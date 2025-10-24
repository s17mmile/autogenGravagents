# --- Imports ---
import numpy as np
import re
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation, welch
from pycbc.waveform import get_fd_waveform
from pycbc.filter import matched_filter

# --- Task 1: Data Loading ---
print("="*60)
print("TASK 1: Downloading strain data for GW170608 (H1 and L1)")
print("="*60)

gps_center = 1180922494.5
duration = 64  # seconds
start = gps_center - duration / 2
end = gps_center + duration / 2

channels = {
    'H1': 'H1:GWOSC-16KHZ_R1_STRAIN',
    'L1': 'L1:GWOSC-16KHZ_R1_STRAIN'
}

strain_H1 = None
strain_L1 = None

def fetch_strain(detector, channel, start, end):
    try:
        print(f"Fetching {detector} strain data from {start} to {end} (GPS)...")
        ts = TimeSeries.get(channel, start, end, cache=True)
        print(f"{detector} data download complete. Duration: {ts.duration.value} seconds, Sample rate: {ts.sample_rate.value} Hz")
        return ts
    except Exception as e:
        print(f"Error fetching {detector} data: {e}")
        return None

strain_H1 = fetch_strain('H1', channels['H1'], start, end)
strain_L1 = fetch_strain('L1', channels['L1'], start, end)

if strain_H1 is None and strain_L1 is None:
    raise RuntimeError("Failed to download strain data for both H1 and L1. Exiting.")

# --- Task 2: Preprocessing ---
print("\n" + "="*60)
print("TASK 2: Preprocessing (Bandpass 30–300 Hz and Whitening)")
print("="*60)

def preprocess_strain(strain, detector_label):
    if strain is None:
        print(f"No data for {detector_label}, skipping preprocessing.")
        return None
    try:
        print(f"Applying bandpass filter (30–300 Hz) to {detector_label} data...")
        strain_bp = strain.bandpass(30, 300, filtfilt=True)
        print(f"Whitening {detector_label} data...")
        strain_whitened = strain_bp.whiten()
        print(f"{detector_label} preprocessing complete.")
        return strain_whitened
    except Exception as e:
        print(f"Error preprocessing {detector_label} data: {e}")
        return None

strain_H1_processed = preprocess_strain(strain_H1, "H1")
strain_L1_processed = preprocess_strain(strain_L1, "L1")

if strain_H1_processed is None and strain_L1_processed is None:
    raise RuntimeError("Preprocessing failed for both H1 and L1. Exiting.")

# --- Task 3: Matched Filtering ---
print("\n" + "="*60)
print("TASK 3: Matched Filtering with IMRPhenomPv2 Template Bank")
print("="*60)

def gwpy_to_pycbc(ts):
    return PyCBC_TimeSeries(ts.value, delta_t=ts.dt.value if hasattr(ts.dt, 'value') else ts.dt, epoch=ts.t0.value if hasattr(ts.t0, 'value') else ts.t0)

# Template bank grid (coarse for demonstration)
m1_vals = np.linspace(5, 20, 4)      # 5, 10, 15, 20
m2_vals = np.linspace(5, 15, 3)      # 5, 10, 15
spin_vals = np.linspace(-0.99, 0.99, 3)  # -0.99, 0.0, 0.99

template_bank = []
for m1 in m1_vals:
    for m2 in m2_vals:
        if m2 > m1:
            continue  # enforce m1 >= m2
        for s1z in spin_vals:
            for s2z in spin_vals:
                template_bank.append({'mass1': m1, 'mass2': m2, 'spin1z': s1z, 'spin2z': s2z})

def run_matched_filter(strain_gwpy, detector_label):
    if strain_gwpy is None:
        print(f"No preprocessed data for {detector_label}, skipping matched filtering.")
        return []
    try:
        print(f"Converting {detector_label} data to PyCBC TimeSeries...")
        strain = gwpy_to_pycbc(strain_gwpy)
        print(f"Computing PSD for {detector_label}...")
        psd = welch(strain, seg_len=4, avg_method='median')
        psd = interpolate(psd, strain.delta_f)
        psd = inverse_spectrum_truncation(psd, int(4 * strain.sample_rate))
        results = []
        print(f"Running matched filtering for {detector_label} ({len(template_bank)} templates)...")
        for i, params in enumerate(template_bank):
            try:
                hp, _ = get_fd_waveform(
                    approximant="IMRPhenomPv2",
                    mass1=params['mass1'],
                    mass2=params['mass2'],
                    spin1z=params['spin1z'],
                    spin2z=params['spin2z'],
                    f_lower=30.0,
                    delta_f=strain.delta_f,
                    f_final=strain.sample_rate / 2
                )
                snr = matched_filter(hp, strain, psd=psd, low_frequency_cutoff=30.0)
                snr = snr.crop(4, 4)
                max_snr = abs(snr).numpy().max()
                max_time = snr.sample_times[np.argmax(abs(snr))]
                results.append({
                    'template': f"m1={params['mass1']},m2={params['mass2']},s1z={params['spin1z']},s2z={params['spin2z']}",
                    'max_snr': max_snr,
                    'max_time': float(max_time)
                })
                if (i+1) % 10 == 0 or i == len(template_bank)-1:
                    print(f"Processed {i+1}/{len(template_bank)} templates for {detector_label}...")
            except Exception as e:
                print(f"Template {params} failed: {e}")
        print(f"Matched filtering complete for {detector_label}.")
        return results
    except Exception as e:
        print(f"Error in matched filtering for {detector_label}: {e}")
        return []

results_H1 = run_matched_filter(strain_H1_processed, "H1")
results_L1 = run_matched_filter(strain_L1_processed, "L1")

if not results_H1 and not results_L1:
    raise RuntimeError("Matched filtering failed for both H1 and L1. Exiting.")

# --- Task 4: Identify Best Template ---
print("\n" + "="*60)
print("TASK 4: Identify Template with Highest SNR")
print("="*60)

def extract_template_params(template_str):
    # Example: "m1=10.0,m2=5.0,s1z=0.0,s2z=0.99"
    pattern = r"m1=([-\d.]+),m2=([-\d.]+),s1z=([-\d.]+),s2z=([-\d.]+)"
    match = re.match(pattern, template_str)
    if match:
        return {
            'mass1': float(match.group(1)),
            'mass2': float(match.group(2)),
            'spin1z': float(match.group(3)),
            'spin2z': float(match.group(4))
        }
    else:
        return {}

def find_best_template(results, detector_label):
    if not results or len(results) == 0:
        print(f"No matched filter results for {detector_label}.")
        return None
    try:
        best = max(results, key=lambda x: x['max_snr'])
        params = extract_template_params(best['template'])
        print(f"\nBest template for {detector_label}:")
        print(f"  SNR: {best['max_snr']:.2f} at time {best['max_time']:.4f} s")
        print(f"  Parameters: mass1={params.get('mass1')}, mass2={params.get('mass2')}, spin1z={params.get('spin1z')}, spin2z={params.get('spin2z')}")
        return {'detector': detector_label, 'snr': best['max_snr'], 'time': best['max_time'], **params}
    except Exception as e:
        print(f"Error identifying best template for {detector_label}: {e}")
        return None

best_H1 = find_best_template(results_H1, "H1")
best_L1 = find_best_template(results_L1, "L1")

print("\nWorkflow complete.")

# Optionally, save results to disk (uncomment if desired)
# import json
# with open("results_H1.json", "w") as f:
#     json.dump(results_H1, f, indent=2)
# with open("results_L1.json", "w") as f:
#     json.dump(results_L1, f, indent=2)