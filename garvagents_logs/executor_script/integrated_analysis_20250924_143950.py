# ============================================================
# GW170608 Gravitational Wave Data Analysis Integrated Script
# ============================================================

# -------------------------
# Imports and Setup
# -------------------------
import os
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.catalog import Merger
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.psd import welch
from pycbc.filter import matched_filter
from pycbc.waveform import get_fd_waveform
from pycbc import pnutils

# -------------------------
# Output Directory
# -------------------------
output_dir = "gw170608_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Task 1: Data Loading
# -------------------------
print("="*60)
print("TASK 1: Downloading H1 and L1 strain data for GW170608")
print("="*60)
try:
    print("Fetching GPS time for GW170608...")
    m = Merger('GW170608')
    gps_time = m.time
    print(f"GW170608 GPS time: {gps_time}")
except Exception as e:
    print(f"Error fetching GPS time: {e}")
    raise

start_time = gps_time - 32
end_time = gps_time + 32

strain_H1 = None
strain_L1 = None

for det in ['H1', 'L1']:
    try:
        print(f"Downloading {det} strain data from {start_time} to {end_time}...")
        ts = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        if det == 'H1':
            strain_H1 = ts
        else:
            strain_L1 = ts
        print(f"{det} data download complete. Duration: {ts.duration.value} seconds.")
        ts.write(os.path.join(output_dir, f"strain_{det}.gwf"))
    except Exception as e:
        print(f"Error downloading {det} data: {e}")

if strain_H1 is None or strain_L1 is None:
    print("Critical error: Could not fetch both H1 and L1 data. Exiting.")
    exit(1)

# -------------------------
# Task 2: Preprocessing
# -------------------------
print("\n" + "="*60)
print("TASK 2: Preprocessing strain data (bandpass 30–300 Hz, whitening)")
print("="*60)

def preprocess_strain(strain, det):
    try:
        print(f"Applying 30–300 Hz bandpass filter to {det} data...")
        strain_bp = strain.bandpass(30, 300)
        print(f"Whitening {det} data...")
        strain_white = strain_bp.whiten()
        print(f"{det} preprocessing complete.")
        return strain_white
    except Exception as e:
        print(f"Error preprocessing {det} data: {e}")
        return None

strain_H1_proc = preprocess_strain(strain_H1, 'H1')
strain_L1_proc = preprocess_strain(strain_L1, 'L1')

if strain_H1_proc is None or strain_L1_proc is None:
    print("Critical error: Preprocessing failed for one or both detectors. Exiting.")
    exit(1)

strain_H1_proc.write(os.path.join(output_dir, "strain_H1_proc.gwf"))
strain_L1_proc.write(os.path.join(output_dir, "strain_L1_proc.gwf"))

# -------------------------
# Task 3: PSD Adjustment
# -------------------------
print("\n" + "="*60)
print("TASK 3: Adjusting PSD estimation parameters for 64-second data segment")
print("="*60)

duration = strain_H1_proc.duration.value  # Should be 64 seconds

if duration >= 64:
    psd_segment_length = 8  # seconds
    psd_overlap = 4         # seconds (50% overlap)
elif duration >= 16:
    psd_segment_length = 4
    psd_overlap = 2
else:
    raise ValueError("Data segment too short for reliable PSD estimation.")

n_segments = (duration - psd_overlap) // (psd_segment_length - psd_overlap)
if n_segments < 2:
    print("Warning: Fewer than 2 PSD segments; consider reducing segment length or overlap.")

print(f"PSD segment length set to {psd_segment_length} seconds.")
print(f"PSD overlap set to {psd_overlap} seconds.")
print(f"Number of PSD segments: {int(n_segments)}")

psd_params = {
    'segment_length': psd_segment_length,
    'overlap': psd_overlap,
    'n_segments': int(n_segments)
}

with open(os.path.join(output_dir, "psd_params.txt"), "w") as f:
    for k, v in psd_params.items():
        f.write(f"{k}: {v}\n")

# -------------------------
# Task 4: Matched Filtering and Parameter Extraction
# -------------------------
print("\n" + "="*60)
print("TASK 4: Matched filtering and parameter extraction (H1 only)")
print("="*60)

def gwpy_to_pycbc(ts):
    return PyCBC_TimeSeries(ts.value, delta_t=ts.dt.value, epoch=ts.t0.value)

try:
    print("Converting GWpy TimeSeries to PyCBC TimeSeries...")
    pycbc_strain = gwpy_to_pycbc(strain_H1_proc)
except Exception as e:
    print(f"Error converting GWpy to PyCBC TimeSeries: {e}")
    exit(1)

try:
    print("Estimating PSD...")
    seglen = psd_params['segment_length']
    overlap = psd_params['overlap']
    psd = welch(pycbc_strain, seg_len=seglen, seg_stride=seglen-overlap)
    psd = psd.interpolate(len(pycbc_strain)//2 + 1)
    psd = psd.trim_freq(30, 300)
    print("PSD estimation complete.")
except Exception as e:
    print(f"Error estimating PSD: {e}")
    exit(1)

print("Generating template bank...")
m1_vals = np.linspace(5, 20, 8)  # 8 values from 5 to 20
m2_vals = np.linspace(5, 15, 5)  # 5 values from 5 to 15
spin_vals = np.linspace(-0.99, 0.99, 3)  # -0.99, 0, 0.99

templates = []
for m1 in m1_vals:
    for m2 in m2_vals:
        if m2 > m1:
            continue  # enforce m1 >= m2
        for spin1z in spin_vals:
            for spin2z in spin_vals:
                templates.append({
                    'mass1': m1,
                    'mass2': m2,
                    'spin1z': spin1z,
                    'spin2z': spin2z,
                    'approximant': 'IMRPhenomPv2',
                    'f_lower': 30
                })
print(f"Template bank size: {len(templates)}")

print("Performing matched filtering...")
max_snr = 0
best_template = None
best_snr_series = None
best_index = None

for idx, params in enumerate(templates):
    try:
        hp, _ = get_fd_waveform(approximant=params['approximant'],
                                mass1=params['mass1'],
                                mass2=params['mass2'],
                                spin1z=params['spin1z'],
                                spin2z=params['spin2z'],
                                delta_f=pycbc_strain.delta_f,
                                f_lower=params['f_lower'])
        hp = hp[:len(psd)]
        snr = matched_filter(hp, pycbc_strain, psd=psd, low_frequency_cutoff=30)
        snr = snr.crop(4, 4)  # Remove edge effects
        peak = abs(snr).numpy().max()
        if peak > max_snr:
            max_snr = peak
            best_template = params
            best_snr_series = snr
            best_index = idx
        if idx % 20 == 0:
            print(f"Processed {idx+1}/{len(templates)} templates...")
    except Exception as e:
        print(f"Template {idx} failed: {e}")

print("Matched filtering complete.")
print(f"Best SNR: {max_snr:.2f} (template index {best_index})")
print(f"Best template parameters: {best_template}")

# Extract physical parameters
result = {}
if best_template is not None:
    result['component_masses'] = (best_template['mass1'], best_template['mass2'])
    result['component_spins'] = (best_template['spin1z'], best_template['spin2z'])
    # Estimate merger time
    merger_time = best_snr_series.sample_times[np.argmax(abs(best_snr_series))]
    result['merger_time'] = merger_time
    # Final mass and spin estimation
    try:
        final_mass, final_spin = pnutils.final_mass_spin(
            best_template['mass1'], best_template['mass2'],
            best_template['spin1z'], best_template['spin2z']
        )
        result['final_mass'] = final_mass
        result['final_spin'] = final_spin
    except Exception as e:
        print(f"Error computing final mass/spin: {e}")
        result['final_mass'] = None
        result['final_spin'] = None
    # Luminosity distance not estimated here
    result['luminosity_distance'] = "Not estimated (requires Bayesian inference)"
    print(f"Merger time (GPS): {merger_time}")
else:
    print("No valid template found.")

# Save results for downstream use
np.save(os.path.join(output_dir, "best_snr_series.npy"),
        best_snr_series.numpy() if best_snr_series is not None else np.array([]))
with open(os.path.join(output_dir, "best_template.txt"), "w") as f:
    for k, v in (best_template or {}).items():
        f.write(f"{k}: {v}\n")
    f.write(f"max_SNR: {max_snr}\n")
with open(os.path.join(output_dir, "extracted_parameters.txt"), "w") as f:
    for k, v in result.items():
        f.write(f"{k}: {v}\n")

print("\nExtracted physical parameters:")
for k, v in result.items():
    print(f"{k}: {v}")

print("\nAnalysis complete. Results saved in:", output_dir)