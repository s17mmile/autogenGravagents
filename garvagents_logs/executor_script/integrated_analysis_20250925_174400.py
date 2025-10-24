# --- Imports ---
import numpy as np
import os
import pickle
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PycbcTimeSeries
from pycbc.filter import matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import utils as wfutils

# --- Section 1: Download LIGO Strain Data ---
print("\n=== 1. Downloading LIGO H1 and L1 strain data for GW170608 ===")
event_gps = 1180922494.5
window = 32  # seconds before and after
start = event_gps - window
end = event_gps + window
detectors = ['H1', 'L1']
strain_data = {}

os.makedirs("results", exist_ok=True)
print(f"Attempting to download strain data for GW170608 (GPS {event_gps})...")

for det in detectors:
    try:
        print(f"Fetching {det} data from {start} to {end}...")
        ts = GwpyTimeSeries.fetch_open_data(det, start, end, cache=True)
        strain_data[det] = ts
        print(f"Successfully downloaded {det} data.")
        # Save raw data
        ts.write(f"results/{det}_raw_strain.gwf", format='gwf')
    except Exception as e:
        print(f"Error fetching {det} data: {e}")
        strain_data[det] = None

# --- Section 2: Preprocessing (Bandpass + Whitening) ---
print("\n=== 2. Preprocessing: Bandpass filtering and whitening ===")
preprocessed_data = {}

print("Starting preprocessing: bandpass filtering (30-300 Hz) and whitening...")

for det, ts in strain_data.items():
    if ts is None:
        print(f"Skipping {det}: no data available.")
        preprocessed_data[det] = None
        continue
    try:
        print(f"Processing {det}...")
        ts_bp = ts.bandpass(30, 300)
        ts_white = ts_bp.whiten()
        preprocessed_data[det] = ts_white
        print(f"{det} preprocessing complete.")
        # Save preprocessed data
        ts_white.write(f"results/{det}_preprocessed_strain.gwf", format='gwf')
    except Exception as e:
        print(f"Error preprocessing {det}: {e}")
        preprocessed_data[det] = None

# --- Section 3: Template Bank Generation ---
print("\n=== 3. Generating template bank (IMRPhenomPv2) ===")
mass1_vals = np.arange(10, 15, 1)  # 10, 11, 12, 13, 14
mass2_vals = np.arange(7, 12, 1)   # 7, 8, 9, 10, 11
spin_vals = [-0.3, 0.0, 0.3]
duration = 16
sample_rate = 4096
f_lower = 30.0
f_upper = 300.0

template_bank = []
template_failures = []

print("Generating template bank...")

for m1 in mass1_vals:
    for m2 in mass2_vals:
        if m2 > m1:
            continue  # enforce m1 >= m2
        for s1z in spin_vals:
            for s2z in spin_vals:
                params = {
                    'mass1': float(m1),
                    'mass2': float(m2),
                    'spin1z': float(s1z),
                    'spin2z': float(s2z),
                    'approximant': 'IMRPhenomPv2',
                    'delta_t': 1.0 / sample_rate,
                    'f_lower': f_lower
                }
                try:
                    hp, _ = get_td_waveform(
                        approximant=params['approximant'],
                        mass1=params['mass1'],
                        mass2=params['mass2'],
                        spin1z=params['spin1z'],
                        spin2z=params['spin2z'],
                        delta_t=params['delta_t'],
                        f_lower=params['f_lower'],
                        duration=duration
                    )
                    # Truncate or pad to exact length
                    target_len = int(duration * sample_rate)
                    if len(hp) > target_len:
                        hp = hp[:target_len]
                    elif len(hp) < target_len:
                        hp = hp.append_zeros(target_len - len(hp))
                    template_bank.append({'params': params, 'hp': hp})
                    print(f"Template generated: m1={m1}, m2={m2}, s1z={s1z}, s2z={s2z}")
                except Exception as e:
                    print(f"Failed: m1={m1}, m2={m2}, s1z={s1z}, s2z={s2z} ({e})")
                    template_failures.append({'params': params, 'error': str(e)})

print(f"\nTemplate bank generation complete.")
print(f"Total templates generated: {len(template_bank)}")
print(f"Total failures: {len(template_failures)}")
with open("results/template_bank.pkl", "wb") as f:
    pickle.dump(template_bank, f)
with open("results/template_bank_failures.pkl", "wb") as f:
    pickle.dump(template_failures, f)

# --- Section 4: Matched Filtering and Analysis ---
print("\n=== 4. Matched filtering and analysis ===")
GW170608_params = {
    'chirp_mass': 7.9,  # solar masses (approximate, from LIGO/Virgo GWTC-1)
    'mass_ratio': 0.67  # m2/m1 (approximate)
}

results = []
filter_failures = []

print("Starting matched filtering for all templates...")

# Helper: Convert GWpy TimeSeries to PyCBC TimeSeries
def gwpy_to_pycbc(ts):
    return PycbcTimeSeries(ts.value, delta_t=ts.dt.value, epoch=ts.t0.value)

# Prepare PyCBC TimeSeries for each detector
pycbc_data = {}
for det in ['H1', 'L1']:
    ts = preprocessed_data.get(det)
    if ts is None:
        pycbc_data[det] = None
        continue
    try:
        pycbc_data[det] = gwpy_to_pycbc(ts)
    except Exception as e:
        print(f"Error converting {det} data to PyCBC TimeSeries: {e}")
        pycbc_data[det] = None

for idx, template in enumerate(template_bank):
    params = template['params']
    hp = template['hp']
    template_result = {'params': params, 'peak_network_snr': None, 'peak_time': None, 'peak_snr_H1': None, 'peak_snr_L1': None}
    try:
        snr_dict = {}
        for det in ['H1', 'L1']:
            data = pycbc_data.get(det)
            if data is None:
                raise ValueError(f"No preprocessed data for {det}")
            # Ensure template and data have same sample rate and length
            if hp.delta_t != data.delta_t:
                raise ValueError(f"Sample rate mismatch: template {1/hp.delta_t} Hz, data {1/data.delta_t} Hz")
            if len(hp) > len(data):
                hp_use = hp[:len(data)]
            elif len(hp) < len(data):
                hp_use = hp.append_zeros(len(data) - len(hp))
            else:
                hp_use = hp
            # Estimate PSD from data
            psd = data.psd(4 * data.sample_rate)
            psd = interpolate(psd, len(data))
            psd = inverse_spectrum_truncation(psd, int(4 * data.sample_rate))
            # Matched filter
            snr = matched_filter(hp_use, data, psd=psd, low_frequency_cutoff=30.0)
            snr = snr.crop(4, 4)  # Remove filter transients
            snr_dict[det] = snr
        # Combine SNRs for network SNR (quadrature sum)
        # Find peak SNR time (use H1 as reference)
        peak_idx = np.argmax(np.abs(snr_dict['H1']))
        peak_time = snr_dict['H1'].sample_times[peak_idx]
        snr_H1 = np.abs(snr_dict['H1'][peak_idx])
        snr_L1 = np.abs(snr_dict['L1'][peak_idx])
        network_snr = np.sqrt(snr_H1**2 + snr_L1**2)
        template_result.update({
            'peak_network_snr': float(network_snr),
            'peak_time': float(peak_time),
            'peak_snr_H1': float(snr_H1),
            'peak_snr_L1': float(snr_L1)
        })
        results.append(template_result)
        if (idx + 1) % 20 == 0 or (idx + 1) == len(template_bank):
            print(f"Processed {idx+1}/{len(template_bank)} templates...")
    except Exception as e:
        print(f"Error in matched filtering for template {idx}: {e}")
        filter_failures.append({'params': params, 'error': str(e)})

print(f"\nMatched filtering complete. {len(results)} templates processed, {len(filter_failures)} failures.")
with open("results/matched_filter_results.pkl", "wb") as f:
    pickle.dump(results, f)
with open("results/matched_filter_failures.pkl", "wb") as f:
    pickle.dump(filter_failures, f)

# Find template with highest network SNR
if results:
    best_template = max(results, key=lambda x: x['peak_network_snr'])
    best_params = best_template['params']
    m1 = best_params['mass1']
    m2 = best_params['mass2']
    chirp_mass = wfutils.chirp_mass(m1, m2)
    mass_ratio = m2 / m1
    print("\n=== Best-fit template parameters ===")
    print(f"  mass1: {m1:.2f} M_sun")
    print(f"  mass2: {m2:.2f} M_sun")
    print(f"  spin1z: {best_params['spin1z']:.2f}")
    print(f"  spin2z: {best_params['spin2z']:.2f}")
    print(f"  Chirp mass: {chirp_mass:.2f} M_sun (GW170608: {GW170608_params['chirp_mass']} M_sun)")
    print(f"  Mass ratio: {mass_ratio:.2f} (GW170608: {GW170608_params['mass_ratio']})")
    print(f"  Peak network SNR: {best_template['peak_network_snr']:.2f}")
    print(f"  Peak time: {best_template['peak_time']:.2f} s")
else:
    print("No successful matched filtering results.")

print("\n=== Template bank statistics ===")
print(f"  Total templates attempted: {len(template_bank)}")
print(f"  Successful templates: {len(results)}")
print(f"  Failed templates: {len(filter_failures)}")
if results:
    print(f"  Highest network SNR: {best_template['peak_network_snr']:.2f}")

print("\nAll intermediate and final results saved in the 'results/' directory.")