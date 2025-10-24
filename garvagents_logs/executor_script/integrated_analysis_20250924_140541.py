# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt

from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design

from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.filter import matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation, welch

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# --- Constants ---
CENTER_GPS = 1180922494.5
INTERVAL = 32  # seconds
START_TIME = CENTER_GPS - INTERVAL
END_TIME = CENTER_GPS + INTERVAL
F_LOWER = 30.0  # Hz
F_UPPER = 300.0  # Hz
PSD_SEG_LEN = 4  # seconds for PSD estimation
TEMPLATE_DELTA_F = 1.0 / 64  # Frequency resolution, assuming 64s data

# --- Task 1: Data Loading ---
print("="*60)
print("TASK 1: Downloading strain data for GW170608 (H1 and L1)")
print("="*60)
strain_H1 = None
strain_L1 = None

try:
    print(f"Fetching H1 strain data from {START_TIME} to {END_TIME} (GPS)...")
    strain_H1 = TimeSeries.get('H1:GWOSC-16KHZ_R1_STRAIN', START_TIME, END_TIME, cache=True)
    print("H1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error downloading H1 strain data: {e}")

try:
    print(f"Fetching L1 strain data from {START_TIME} to {END_TIME} (GPS)...")
    strain_L1 = TimeSeries.get('L1:GWOSC-16KHZ_R1_STRAIN', START_TIME, END_TIME, cache=True)
    print("L1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error downloading L1 strain data: {e}")

if strain_H1 is None and strain_L1 is None:
    raise RuntimeError("Failed to download strain data for both H1 and L1. Exiting.")

# --- Task 2: Preprocessing ---
print("\n" + "="*60)
print("TASK 2: Preprocessing (Bandpass 30–300 Hz and Whitening)")
print("="*60)

def preprocess_strain(strain, ifo_label):
    """Apply bandpass filter and whitening to a GWpy TimeSeries."""
    if strain is None:
        print(f"No data available for {ifo_label}. Skipping preprocessing.")
        return None
    try:
        print(f"Applying {F_LOWER}–{F_UPPER} Hz Butterworth bandpass filter to {ifo_label} data...")
        strain_bp = strain.bandpass(F_LOWER, F_UPPER, filtfilt=True)
        print(f"Bandpass filter applied to {ifo_label}.")
        print(f"Whitening {ifo_label} data...")
        strain_white = strain_bp.whiten()
        print(f"Whitening complete for {ifo_label}.")
        return strain_white
    except Exception as e:
        print(f"Error preprocessing {ifo_label} data: {e}")
        return None

strain_H1_processed = preprocess_strain(strain_H1, "H1")
strain_L1_processed = preprocess_strain(strain_L1, "L1")

if strain_H1_processed is None and strain_L1_processed is None:
    raise RuntimeError("Preprocessing failed for both H1 and L1. Exiting.")

# --- Task 3: Matched Filtering ---
print("\n" + "="*60)
print("TASK 3: Matched Filtering with IMRPhenomPv2 Template Bank")
print("="*60)

# Helper: Convert GWpy TimeSeries to PyCBC TimeSeries
def gwpy_to_pycbc(ts):
    return PyCBC_TimeSeries(ts.value, delta_t=ts.dt.value if hasattr(ts.dt, 'value') else ts.dt, epoch=ts.t0.value if hasattr(ts.t0, 'value') else ts.t0)

# Template bank parameters (coarse grid for demonstration)
mass1_range = np.linspace(5, 20, 4)   # Msun
mass2_range = np.linspace(5, 15, 3)   # Msun
spin_range = np.linspace(-0.99, 0.99, 2)  # -0.99, 0.99

def run_matched_filter(strain, ifo_label):
    if strain is None:
        print(f"No preprocessed data for {ifo_label}. Skipping matched filtering.")
        return []
    try:
        print(f"Converting {ifo_label} data to PyCBC TimeSeries...")
        strain_pycbc = gwpy_to_pycbc(strain)
        print(f"Computing PSD for {ifo_label}...")
        psd = welch(strain_pycbc, seg_len=PSD_SEG_LEN, avg_method='median')
        psd = interpolate(psd, strain_pycbc.delta_f)
        psd = inverse_spectrum_truncation(psd, int(4 * strain_pycbc.sample_rate))
        print(f"PSD computed for {ifo_label}.")

        print(f"Generating template bank for {ifo_label}...")
        templates = []
        for m1 in mass1_range:
            for m2 in mass2_range:
                if m2 > m1:
                    continue  # enforce m1 >= m2
                for spin1z in spin_range:
                    for spin2z in spin_range:
                        try:
                            hp, _ = get_fd_waveform(
                                approximant="IMRPhenomPv2",
                                mass1=m1, mass2=m2,
                                spin1z=spin1z, spin2z=spin2z,
                                f_lower=F_LOWER,
                                delta_f=strain_pycbc.delta_f,
                                f_final=strain_pycbc.sample_rate / 2
                            )
                            templates.append((f"m1={m1},m2={m2},s1z={spin1z},s2z={spin2z}", hp))
                        except Exception as e:
                            print(f"Template generation failed for m1={m1}, m2={m2}, s1z={spin1z}, s2z={spin2z}: {e}")
        print(f"Template bank generated for {ifo_label}: {len(templates)} templates.")

        print(f"Running matched filtering for {ifo_label}...")
        triggers = []
        for label, template in templates:
            try:
                snr = matched_filter(template, strain_pycbc, psd=psd, low_frequency_cutoff=F_LOWER)
                snr = snr.crop(4, 4)  # Remove filter wraparound
                max_snr = abs(snr).numpy().max()
                triggers.append({'template': label, 'max_snr': max_snr})
                print(f"Template {label}: max SNR = {max_snr:.2f}")
            except Exception as e:
                print(f"Matched filtering failed for template {label}: {e}")
        print(f"Matched filtering complete for {ifo_label}.")
        return triggers
    except Exception as e:
        print(f"Error during matched filtering for {ifo_label}: {e}")
        return []

results_H1 = run_matched_filter(strain_H1_processed, "H1")
results_L1 = run_matched_filter(strain_L1_processed, "L1")

if not results_H1 and not results_L1:
    raise RuntimeError("Matched filtering failed for both H1 and L1. Exiting.")

# --- Task 4: Visualization ---
print("\n" + "="*60)
print("TASK 4: Visualization of Matched Filter SNR Time Series")
print("="*60)

GW170608_GPS = CENTER_GPS

# Helper: Find best template from results
def get_best_template(results):
    if not results:
        return None
    return max(results, key=lambda x: x['max_snr'])

# Helper: Parse template label string
def parse_template_label(label):
    # label format: "m1=...,m2=...,s1z=...,s2z=..."
    parts = label.split(',')
    params = {}
    for part in parts:
        k, v = part.split('=')
        params[k.strip()] = float(v)
    return {
        'mass1': params['m1'],
        'mass2': params['m2'],
        'spin1z': params['s1z'],
        'spin2z': params['s2z']
    }

# Helper: Recompute SNR time series for best template
def compute_snr_series(strain, template_params, psd, f_lower):
    try:
        hp, _ = get_fd_waveform(
            approximant="IMRPhenomPv2",
            mass1=template_params['mass1'],
            mass2=template_params['mass2'],
            spin1z=template_params['spin1z'],
            spin2z=template_params['spin2z'],
            f_lower=f_lower,
            delta_f=strain.delta_f,
            f_final=strain.sample_rate / 2
        )
        snr = matched_filter(hp, strain, psd=psd, low_frequency_cutoff=f_lower)
        snr = snr.crop(4, 4)
        return snr
    except Exception as e:
        print(f"Error computing SNR series: {e}")
        return None

# Choose detector for visualization (H1 preferred if available)
detector_label = None
results = None
strain_processed = None

if results_H1:
    detector_label = 'H1'
    results = results_H1
    strain_processed = strain_H1_processed
elif results_L1:
    detector_label = 'L1'
    results = results_L1
    strain_processed = strain_L1_processed
else:
    print("No matched filter results available for visualization.")
    results = []
    strain_processed = None

if results and strain_processed is not None:
    print(f"Visualizing matched filter SNR for {detector_label}...")

    strain_pycbc = gwpy_to_pycbc(strain_processed)

    # Compute PSD
    print("Computing PSD for visualization...")
    psd = welch(strain_pycbc, seg_len=PSD_SEG_LEN, avg_method='median')
    psd = interpolate(psd, strain_pycbc.delta_f)
    psd = inverse_spectrum_truncation(psd, int(4 * strain_pycbc.sample_rate))

    # Find best template
    best = get_best_template(results)
    if best is None:
        print("No best template found.")
    else:
        print(f"Best template: {best['template']} (max SNR={best['max_snr']:.2f})")
        template_params = parse_template_label(best['template'])

        # Compute SNR time series for best template
        snr = compute_snr_series(strain_pycbc, template_params, psd, f_lower=F_LOWER)
        if snr is None:
            print("Could not compute SNR time series for visualization.")
        else:
            # Find peak SNR and its time
            peak_idx = np.argmax(abs(snr))
            peak_time = snr.sample_times[peak_idx]
            peak_snr = abs(snr[peak_idx])

            # Plot SNR time series
            plt.figure(figsize=(10, 6))
            plt.plot(snr.sample_times, abs(snr), label='Matched Filter SNR')
            plt.axvline(GW170608_GPS, color='r', linestyle='--', label='GW170608 Event Time')
            plt.axvline(peak_time, color='g', linestyle=':', label=f'Peak SNR ({peak_snr:.2f})')
            plt.xlabel('GPS Time (s)')
            plt.ylabel('SNR')
            plt.title(f'{detector_label} Matched Filter SNR Time Series\nBest Template: {best["template"]}')
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Overlay whitened data and template (in time domain)
            try:
                print("Overlaying whitened data and best-fit template (optional)...")
                hp_td, _ = get_td_waveform(
                    approximant="IMRPhenomPv2",
                    mass1=template_params['mass1'],
                    mass2=template_params['mass2'],
                    spin1z=template_params['spin1z'],
                    spin2z=template_params['spin2z'],
                    delta_t=strain_pycbc.delta_t,
                    f_lower=F_LOWER
                )
                # Align template to peak
                template_time = hp_td.start_time + np.argmax(abs(hp_td))
                shift = peak_time - template_time
                hp_td = hp_td.cyclic_time_shift(shift)
                # Whiten template
                hp_td = hp_td.whiten(4, 4, psd, low_frequency_cutoff=F_LOWER)
                # Extract segment around event
                idx0 = int((GW170608_GPS - 1 - strain_pycbc.start_time) / strain_pycbc.delta_t)
                idx1 = int((GW170608_GPS + 1 - strain_pycbc.start_time) / strain_pycbc.delta_t)
                tvec = strain_pycbc.sample_times[idx0:idx1]
                plt.figure(figsize=(10, 6))
                plt.plot(tvec, strain_pycbc[idx0:idx1], label='Whitened Data')
                plt.plot(tvec, hp_td[:len(tvec)], label='Best-fit Template', alpha=0.7)
                plt.axvline(GW170608_GPS, color='r', linestyle='--', label='GW170608 Event Time')
                plt.xlabel('GPS Time (s)')
                plt.ylabel('Whitened Strain')
                plt.title(f'{detector_label} Whitened Data and Best-fit Template')
                plt.legend()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not overlay whitened data and template: {e}")
else:
    print("No data available for visualization.")

print("\nWorkflow complete.")