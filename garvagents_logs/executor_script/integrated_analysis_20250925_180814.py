--------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================
# GW170608 LIGO Data Analysis Pipeline
# ==============================

# --- Imports ---
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from scipy.signal import welch, get_window
from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter
from pycbc.types import TimeSeries as PyCBC_TimeSeries
import traceback
from collections import Counter
import pickle
import os

# ==============================
# 1. Data Loading & Validation
# ==============================
print("\n========== 1. Data Loading & Validation ==========")

# Constants for GW170608
event_gps = 1180922494.5
window = 32  # seconds
start = event_gps - window
end = event_gps + window
detectors = ['H1', 'L1']

strain_data = {}
valid_segments = {}

for det in detectors:
    print(f"\n--- Processing {det} ---")
    try:
        print(f"Fetching open data for {det} from {start} to {end}...")
        ts = TimeSeries.fetch_open_data(det, start, end, cache=True)
        print(f"Data fetched: {ts}")
    except Exception as e:
        print(f"ERROR: Failed to fetch data for {det}: {e}")
        strain_data[det] = None
        valid_segments[det] = []
        continue

    # Check for NaN or infinite values
    data = ts.value
    nan_mask = np.isnan(data)
    inf_mask = np.isinf(data)
    bad_mask = nan_mask | inf_mask
    n_bad = np.sum(bad_mask)
    print(f"Found {n_bad} NaN or infinite samples in {det} data.")

    # Find continuous good segments of at least 60s
    sample_rate = ts.sample_rate.value
    min_samples = int(60 * sample_rate)
    print(f"Sample rate: {sample_rate} Hz, minimum samples for 60s: {min_samples}")

    good_mask = ~bad_mask

    def find_good_segments(mask, min_len):
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1 if mask[0] == False else np.array([0])
        stops = np.where(diff == -1)[0] + 1 if mask[-1] == True else np.array([len(mask)])
        if mask[0]:
            starts = np.insert(starts, 0, 0)
        if mask[-1]:
            stops = np.append(stops, len(mask))
        segments = []
        for s, e in zip(starts, stops):
            if e - s >= min_len:
                segments.append((s, e))
        return segments

    good_segments = find_good_segments(good_mask, min_samples)
    print(f"Found {len(good_segments)} continuous good segments of at least 60s in {det}.")

    segment_times = []
    for s_idx, e_idx in good_segments:
        seg_start = ts.times.value[s_idx]
        seg_end = ts.times.value[e_idx-1]
        segment_times.append((seg_start, seg_end))
        print(f"  Segment: {seg_start:.2f} - {seg_end:.2f} (duration: {seg_end-seg_start:.2f}s)")

    if len(segment_times) == 0:
        print(f"WARNING: No valid 60s segment found for {det}. Skipping further analysis for this detector.")
        strain_data[det] = None
        valid_segments[det] = []
        continue

    strain_data[det] = ts
    valid_segments[det] = segment_times

print("\n--- Data loading and validation complete ---")

# Save intermediate results
with open("strain_data.pkl", "wb") as f:
    pickle.dump({'strain_data': strain_data, 'valid_segments': valid_segments}, f)

# ==============================
# 2. Filtering, PSD, Whitening
# ==============================
print("\n========== 2. Filtering, PSD Estimation, Whitening ==========")

f_low = 30
f_high = 300
welch_segment_lengths = [4, 2]  # seconds
fftlength = 4096
overlap = 0.5

filtered_data = {}
psds = {}
whitened_data = {}

for det in detectors:
    print(f"\n--- Processing {det} ---")
    ts = strain_data.get(det)
    segments = valid_segments.get(det, [])
    if ts is None or not segments:
        print(f"Skipping {det}: No valid data or segments.")
        filtered_data[det] = None
        psds[det] = None
        whitened_data[det] = None
        continue

    seg_start, seg_end = segments[0]
    print(f"Using segment: {seg_start:.2f} - {seg_end:.2f} (duration: {seg_end-seg_start:.2f}s)")
    try:
        ts_seg = ts.crop(seg_start, seg_end)
        print(f"Cropped to segment: {ts_seg}")
    except Exception as e:
        print(f"ERROR: Could not crop {det} data: {e}")
        filtered_data[det] = None
        psds[det] = None
        whitened_data[det] = None
        continue

    try:
        ts_filt = ts_seg.bandpass(f_low, f_high)
        print(f"Applied bandpass filter ({f_low}-{f_high} Hz).")
    except Exception as e:
        print(f"ERROR: Bandpass filtering failed for {det}: {e}")
        filtered_data[det] = None
        psds[det] = None
        whitened_data[det] = None
        continue

    filtered_data[det] = ts_filt

    # PSD estimation (Welch)
    psd = None
    for seglen in welch_segment_lengths:
        nperseg = int(seglen * ts_filt.sample_rate.value)
        noverlap = int(nperseg * overlap)
        print(f"Trying Welch PSD: segment={seglen}s, nperseg={nperseg}, noverlap={noverlap}")
        try:
            freqs, pxx = welch(
                ts_filt.value,
                fs=ts_filt.sample_rate.value,
                window=get_window('hann', nperseg),
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=fftlength if fftlength <= len(ts_filt.value) else len(ts_filt.value),
                detrend='constant',
                scaling='density',
                return_onesided=True
            )
            if np.all(pxx > 0):
                psd = (freqs, pxx)
                print(f"Welch PSD succeeded with {seglen}s segments.")
                break
            else:
                print(f"PSD has non-positive values with {seglen}s segments. Trying fallback...")
        except Exception as e:
            print(f"Welch PSD failed with {seglen}s segments: {e}")

    if psd is None:
        print("Falling back to analytical (flat) PSD.")
        try:
            var = np.var(ts_filt.value)
            freqs = np.fft.rfftfreq(len(ts_filt.value), d=1.0/ts_filt.sample_rate.value)
            pxx = np.full_like(freqs, var)
            if np.all(pxx > 0):
                psd = (freqs, pxx)
                print("Analytical PSD succeeded.")
            else:
                print("Analytical PSD failed (non-positive values).")
                psd = None
        except Exception as e:
            print(f"Analytical PSD computation failed: {e}")
            psd = None

    if psd is None:
        print(f"ERROR: No valid PSD for {det}. Whitening will not be attempted.")
        psds[det] = None
        whitened_data[det] = None
        continue

    psds[det] = psd

    # Whitening
    try:
        freqseries = FrequencySeries(psd[1], frequencies=psd[0], unit=ts_filt.unit**2/ts_filt.unit)
        ts_white = ts_filt.whiten(asd=freqseries)
        print("Whitening succeeded.")
        whitened_data[det] = ts_white
    except Exception as e:
        print(f"Whitening failed for {det}: {e}")
        whitened_data[det] = None

print("\n--- Filtering, PSD estimation, and whitening complete ---")

# Save intermediate results
with open("filtered_psd_whitened.pkl", "wb") as f:
    pickle.dump({'filtered_data': filtered_data, 'psds': psds, 'whitened_data': whitened_data}, f)

# ==============================
# 3. Template Bank Generation
# ==============================
print("\n========== 3. Template Bank Generation ==========")

primary_masses = np.arange(10, 15, 1)
secondary_masses = np.arange(7, 12, 1)
aligned_spins = [-0.3, 0.0, 0.3]
approximant = "IMRPhenomPv2"
f_lower = 30
f_upper = 300
durations = [16, 8]

template_bank = []
template_params = []
template_errors = []

for m1 in primary_masses:
    for m2 in secondary_masses:
        if m2 > m1:
            continue
        for spin in aligned_spins:
            params = {
                'mass1': float(m1),
                'mass2': float(m2),
                'spin1z': float(spin),
                'spin2z': float(spin),
                'approximant': approximant,
                'f_lower': f_lower,
                'f_final': f_upper,
            }
            success = False
            for duration in durations:
                try:
                    hp, hc = get_td_waveform(
                        mass1=params['mass1'],
                        mass2=params['mass2'],
                        spin1z=params['spin1z'],
                        spin2z=params['spin2z'],
                        approximant=params['approximant'],
                        f_lower=params['f_lower'],
                        f_final=params['f_final'],
                        delta_t=1.0/4096,
                        duration=duration
                    )
                    if len(hp) < int(duration * 4096 * 0.8):
                        raise ValueError(f"Waveform too short: {len(hp)} samples for {duration}s")
                    template_bank.append((hp, hc))
                    template_params.append({
                        **params,
                        'duration': duration
                    })
                    print(f"Template generated: m1={m1}, m2={m2}, spin={spin}, duration={duration}s")
                    success = True
                    break
                except Exception as e:
                    print(f"Failed for m1={m1}, m2={m2}, spin={spin}, duration={duration}s: {e}")
                    template_errors.append({
                        'params': {**params, 'duration': duration},
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
            if not success:
                print(f"Skipped: m1={m1}, m2={m2}, spin={spin} (all durations failed)")

print(f"\n--- Template bank generation complete ---")
print(f"Total successful templates: {len(template_bank)}")
print(f"Total failed attempts: {len(template_errors)}")

# Save template bank and errors
with open("template_bank.pkl", "wb") as f:
    pickle.dump({'template_bank': template_bank, 'template_params': template_params, 'template_errors': template_errors}, f)

# ==============================
# 4. Matched Filtering & Reporting
# ==============================
print("\n========== 4. Matched Filtering & Reporting ==========")

# Prepare processed data for PyCBC (use whitened if available, else filtered)
pycbc_data = {}
for det in detectors:
    ts = whitened_data.get(det) if whitened_data.get(det) is not None else filtered_data.get(det)
    if ts is not None:
        try:
            pycbc_data[det] = PyCBC_TimeSeries(ts.value, delta_t=ts.dt.value, epoch=ts.t0.value)
            print(f"{det}: Data prepared for matched filtering (length={len(pycbc_data[det])}, delta_t={pycbc_data[det].delta_t})")
        except Exception as e:
            print(f"{det}: ERROR converting data for PyCBC: {e}")
            pycbc_data[det] = None
    else:
        pycbc_data[det] = None
        print(f"{det}: No valid data for matched filtering.")

mf_results = []
mf_errors = []

print("\n--- Starting matched filtering for all templates ---")
for idx, (template, params) in enumerate(zip(template_bank, template_params)):
    template_success = True
    template_snr = {}
    template_peak = {}
    error_info = {'template_idx': idx, 'params': params, 'errors': {}}
    for det in detectors:
        data = pycbc_data.get(det)
        if data is None:
            error_info['errors'][det] = "No valid data"
            template_success = False
            continue
        try:
            hp = template[0]
            if abs(hp.delta_t - data.delta_t) > 1e-10:
                hp = hp.resample(data.delta_t)
            if len(hp) > len(data):
                hp = hp[:len(data)]
            elif len(hp) < len(data):
                hp = hp.append_zeros(len(data) - len(hp))
            snr = matched_filter(hp, data, psd=None, low_frequency_cutoff=params['f_lower'])
            template_snr[det] = snr
            peak = abs(snr).numpy().max()
            template_peak[det] = peak
            print(f"Template {idx} ({params['mass1']},{params['mass2']},spin={params['spin1z']}), {det}: peak SNR={peak:.2f}")
        except Exception as e:
            tb = traceback.format_exc()
            error_info['errors'][det] = f"{str(e)}\n{tb}"
            template_success = False
    if template_success and all(det in template_peak for det in detectors):
        net_snr = np.sqrt(sum(template_peak[det]**2 for det in detectors))
        mf_results.append({
            'idx': idx,
            'params': params,
            'peak_snr': template_peak,
            'network_snr': net_snr,
            'snr_series': template_snr
        })
    else:
        mf_errors.append(error_info)

print("\n--- Matched filtering complete ---")
print(f"Total templates processed: {len(template_bank)}")
print(f"  Successes: {len(mf_results)}")
print(f"  Failures: {len(mf_errors)}")

# Identify template with highest network SNR
if mf_results:
    best_idx = np.argmax([r['network_snr'] for r in mf_results])
    best_result = mf_results[best_idx]
    print("\n--- Peak Template ---")
    print(f"Template index: {best_result['idx']}")
    print(f"Parameters: {best_result['params']}")
    print(f"Peak SNRs: {best_result['peak_snr']}")
    print(f"Network SNR: {best_result['network_snr']:.2f}")
else:
    print("No successful matched filtering results.")

# Summarize error statistics
error_reasons = []
for err in mf_errors:
    for det, reason in err['errors'].items():
        error_reasons.append(f"{det}: {reason.splitlines()[0]}")
reason_counts = Counter(error_reasons)
print("\n--- Error Statistics ---")
for reason, count in reason_counts.items():
    print(f"{reason}: {count} failures")

# Save final results
with open("matched_filtering_results.pkl", "wb") as f:
    pickle.dump({'mf_results': mf_results, 'mf_errors': mf_errors}, f)

print("\n========== Pipeline Complete ==========")
print("Intermediate and final results saved as .pkl files.")