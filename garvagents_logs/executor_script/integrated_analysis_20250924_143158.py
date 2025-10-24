# ============================================================
# GW170608 Gravitational Wave Data Analysis Integrated Script
# ============================================================

# -------------------------
# Imports and Setup
# -------------------------
import os
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.psd import welch, interpolate
from pycbc.filter import matched_filter, highpass
from pycbc.waveform import get_td_waveform
from pycbc import pnutils

# -------------------------
# Parameters and Paths
# -------------------------
center_gps = 1180922494.5
window = 32  # seconds
start_time = center_gps - window
end_time = center_gps + window

output_dir = "gw170608_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Task 1: Data Loading
# -------------------------
print("="*60)
print("TASK 1: Downloading H1 and L1 strain data for GW170608")
print("="*60)
strain_H1 = None
strain_L1 = None

try:
    print(f"Fetching H1 strain data from {start_time} to {end_time} (GPS)...")
    strain_H1 = TimeSeries.fetch_open_data('H1', start_time, end_time, cache=True)
    print("H1 strain data fetched successfully.")
    strain_H1.write(os.path.join(output_dir, "strain_H1.gwf"))
except Exception as e:
    print(f"Error fetching H1 data: {e}")

try:
    print(f"Fetching L1 strain data from {start_time} to {end_time} (GPS)...")
    strain_L1 = TimeSeries.fetch_open_data('L1', start_time, end_time, cache=True)
    print("L1 strain data fetched successfully.")
    strain_L1.write(os.path.join(output_dir, "strain_L1.gwf"))
except Exception as e:
    print(f"Error fetching L1 data: {e}")

if strain_H1 is None or strain_L1 is None:
    print("Critical error: Could not fetch both H1 and L1 data. Exiting.")
    exit(1)

# -------------------------
# Task 2: Preprocessing
# -------------------------
print("\n" + "="*60)
print("TASK 2: Preprocessing strain data (bandpass 30–300 Hz, whitening)")
print("="*60)

def preprocess_strain(strain, det_label):
    """Apply bandpass filter and whitening to a GWpy TimeSeries."""
    try:
        print(f"Applying 30–300 Hz bandpass filter to {det_label} data...")
        strain_bp = strain.bandpass(30, 300)
        print(f"Bandpass filter applied to {det_label}. Whitening data...")
        strain_whitened = strain_bp.whiten()
        print(f"Whitening complete for {det_label}.")
        return strain_whitened
    except Exception as e:
        print(f"Error preprocessing {det_label} data: {e}")
        return None

strain_H1_proc = preprocess_strain(strain_H1, 'H1')
strain_L1_proc = preprocess_strain(strain_L1, 'L1')

if strain_H1_proc is None or strain_L1_proc is None:
    print("Critical error: Preprocessing failed for one or both detectors. Exiting.")
    exit(1)

# Save preprocessed data
strain_H1_proc.write(os.path.join(output_dir, "strain_H1_proc.gwf"))
strain_L1_proc.write(os.path.join(output_dir, "strain_L1_proc.gwf"))

# -------------------------
# Task 3: Matched Filtering
# -------------------------
print("\n" + "="*60)
print("TASK 3: Matched filtering with IMRPhenomPv2 template bank (H1 only)")
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
    print("Applying highpass filter at 15 Hz...")
    pycbc_strain = highpass(pycbc_strain, 15.0)
except Exception as e:
    print(f"Error applying highpass filter: {e}")
    exit(1)

try:
    print("Estimating PSD using Welch method...")
    psd_length = 4  # seconds
    # Correct calculation: seg_len is the number of samples per segment
    seg_len = int(psd_length / pycbc_strain.delta_t)
    # Ensure seg_len is at least 2 and less than the length of the data
    if seg_len < 2 or seg_len > len(pycbc_strain):
        raise ValueError(f"Invalid seg_len={seg_len} for PSD estimation. Data length: {len(pycbc_strain)}")
    # Ensure seg_len is a power of 2 for FFT efficiency
    def next_lower_power_of_two(n):
        return 2**(n.bit_length()-1)
    seg_len_pow2 = next_lower_power_of_two(seg_len)
    if seg_len_pow2 < 2:
        seg_len_pow2 = 2
    if seg_len_pow2 > len(pycbc_strain):
        seg_len_pow2 = next_lower_power_of_two(len(pycbc_strain))
    print(f"Using seg_len={seg_len_pow2} samples for Welch PSD estimation.")
    psd = welch(pycbc_strain, seg_len=seg_len_pow2, avg_method='median')
    psd = interpolate(psd, pycbc_strain.delta_f)
except Exception as e:
    print(f"Error estimating PSD: {e}")
    exit(1)

# Template bank parameters (coarse grid for demonstration)
m1_vals = np.linspace(5, 20, 4)   # 4 values between 5 and 20
m2_vals = np.linspace(5, 15, 3)   # 3 values between 5 and 15
spin_vals = np.linspace(-0.99, 0.99, 2)  # -0.99, 0.99

snr_max = 0
best_params = None
best_snr_series = None

print("Starting matched filtering over template bank...")
for m1 in m1_vals:
    for m2 in m2_vals:
        if m2 > m1:
            continue  # enforce m1 >= m2
        for spin1z in spin_vals:
            for spin2z in spin_vals:
                try:
                    # Generate template waveform
                    hp, _ = get_td_waveform(approximant="IMRPhenomPv2",
                                            mass1=m1, mass2=m2,
                                            spin1z=spin1z, spin2z=spin2z,
                                            delta_t=pycbc_strain.delta_t,
                                            f_lower=30)
                    # Resize template to match data length
                    hp = hp.crop(0, 0)
                    if len(hp) > len(pycbc_strain):
                        hp = hp[:len(pycbc_strain)]
                    elif len(hp) < len(pycbc_strain):
                        hp = hp.append_zeros(len(pycbc_strain) - len(hp))
                    
                    # Matched filter
                    snr = matched_filter(hp, pycbc_strain, psd=psd, low_frequency_cutoff=30)
                    snr_peak = abs(snr).max()
                    print(f"Template m1={m1:.1f}, m2={m2:.1f}, spin1z={spin1z:.2f}, spin2z={spin2z:.2f}: max SNR={snr_peak:.2f}")
                    
                    if snr_peak > snr_max:
                        snr_max = snr_peak
                        best_params = {'m1': m1, 'm2': m2, 'spin1z': spin1z, 'spin2z': spin2z}
                        best_snr_series = snr
                except Exception as e:
                    print(f"Error with template m1={m1}, m2={m2}, spin1z={spin1z}, spin2z={spin2z}: {e}")

print("\nMatched filtering complete.")
if best_params is not None:
    print(f"Best-fit template: {best_params} with max SNR={snr_max:.2f}")
else:
    print("No valid templates found. Exiting.")
    exit(1)

# Save best SNR time series and parameters
np.save(os.path.join(output_dir, "best_snr_series.npy"), best_snr_series.numpy())
with open(os.path.join(output_dir, "best_params.txt"), "w") as f:
    for k, v in best_params.items():
        f.write(f"{k}: {v}\n")
    f.write(f"max_SNR: {snr_max}\n")

# -------------------------
# Task 4: Parameter Extraction
# -------------------------
print("\n" + "="*60)
print("TASK 4: Extracting physical parameters from best-fit template")
print("="*60)

# Extract component masses and spins
m1 = best_params['m1']
m2 = best_params['m2']
spin1z = best_params['spin1z']
spin2z = best_params['spin2z']

# Compute final black hole mass and spin using PyCBC's pnutils
try:
    final_mass, final_spin = pnutils.final_mass_spin(m1, m2, spin1z, spin2z)
    print(f"Final black hole mass: {final_mass:.2f} Msun, final spin: {final_spin:.3f}")
except Exception as e:
    print(f"Error computing final mass and spin: {e}")
    final_mass, final_spin = None, None

# Luminosity distance: Not estimated from matched filtering alone
luminosity_distance = "Not estimated (requires Bayesian parameter estimation)"

# Merger time: time of peak SNR
try:
    peak_idx = abs(best_snr_series).numpy().argmax()
    merger_time = best_snr_series.sample_times[peak_idx]
    print(f"Merger time (GPS): {merger_time}")
except Exception as e:
    print(f"Error extracting merger time: {e}")
    merger_time = None

# Collect results
extracted_parameters = {
    'component_mass1': m1,
    'component_mass2': m2,
    'spin1z': spin1z,
    'spin2z': spin2z,
    'final_mass': final_mass,
    'final_spin': final_spin,
    'luminosity_distance': luminosity_distance,
    'merger_time': merger_time
}

# Save extracted parameters
with open(os.path.join(output_dir, "extracted_parameters.txt"), "w") as f:
    for k, v in extracted_parameters.items():
        f.write(f"{k}: {v}\n")

print("\nExtracted physical parameters:")
for k, v in extracted_parameters.items():
    print(f"{k}: {v}")

print("\nAnalysis complete. Results saved in:", output_dir)