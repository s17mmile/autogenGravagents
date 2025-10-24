# =========================
# GW170608 Gravitational Wave Analysis Pipeline
# =========================

# ---- Imports ----
import os
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import matched_filter
from pycbc.waveform import get_fd_waveform, get_bank
from pycbc.pnutils import final_mass_spin
import pickle

# ---- Configuration ----
CENTER_GPS = 1180922494.5
DURATION = 64  # seconds
START_TIME = CENTER_GPS - DURATION / 2
BANDPASS_LOW = 30
BANDPASS_HIGH = 300
PSD_SEGMENT_LENGTH = 4  # seconds
TEMPLATE_BANK_PARAMS = {
    "min-mass1": 5.0,
    "max-mass1": 20.0,
    "min-mass2": 5.0,
    "max-mass2": 15.0,
    "min-spin1z": -0.99,
    "max-spin1z": 0.99,
    "min-spin2z": -0.99,
    "max-spin2z": 0.99,
    "approximant": "IMRPhenomPv2",
    "f_lower": 30.0,
    "duration": 4.0,
}
RESULTS_DIR = "gw170608_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# Task 1: Data Loading
# =========================
print("\n=== Task 1: Downloading H1 and L1 strain data for GW170608 ===")
h1_strain = None
l1_strain = None

try:
    print(f"Fetching H1 strain data from GPS {START_TIME} for {DURATION} seconds...")
    h1_strain = TimeSeries.fetch_open_data('H1', START_TIME, DURATION)
    print("H1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error downloading H1 strain data: {e}")

try:
    print(f"Fetching L1 strain data from GPS {START_TIME} for {DURATION} seconds...")
    l1_strain = TimeSeries.fetch_open_data('L1', START_TIME, DURATION)
    print("L1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error downloading L1 strain data: {e}")

if h1_strain is None or l1_strain is None:
    raise RuntimeError("Failed to download both H1 and L1 strain data. Exiting.")

# Save raw data for reproducibility
h1_strain.write(os.path.join(RESULTS_DIR, "h1_strain.gwf"), format='gwf')
l1_strain.write(os.path.join(RESULTS_DIR, "l1_strain.gwf"), format='gwf')

# =========================
# Task 2: Preprocessing
# =========================
print("\n=== Task 2: Preprocessing (bandpass filtering and whitening) ===")
h1_strain_processed = None
l1_strain_processed = None

try:
    print("Applying bandpass filter (30–300 Hz) to H1 data...")
    h1_filtered = h1_strain.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
    print("Whitening H1 data...")
    h1_strain_processed = h1_filtered.whiten()
    print("H1 data preprocessing complete.")
except Exception as e:
    print(f"Error preprocessing H1 data: {e}")

try:
    print("Applying bandpass filter (30–300 Hz) to L1 data...")
    l1_filtered = l1_strain.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
    print("Whitening L1 data...")
    l1_strain_processed = l1_filtered.whiten()
    print("L1 data preprocessing complete.")
except Exception as e:
    print(f"Error preprocessing L1 data: {e}")

if h1_strain_processed is None or l1_strain_processed is None:
    raise RuntimeError("Failed to preprocess both H1 and L1 strain data. Exiting.")

# Save preprocessed data
h1_strain_processed.write(os.path.join(RESULTS_DIR, "h1_strain_processed.gwf"), format='gwf')
l1_strain_processed.write(os.path.join(RESULTS_DIR, "l1_strain_processed.gwf"), format='gwf')

# =========================
# Task 3: Matched Filtering
# =========================
print("\n=== Task 3: Matched Filtering with PyCBC ===")

# Convert GWpy TimeSeries to PyCBC TimeSeries (use H1 for demonstration)
try:
    print("Converting GWpy TimeSeries to PyCBC TimeSeries...")
    h1_data = PyCBC_TimeSeries(h1_strain_processed.value, delta_t=h1_strain_processed.dt.value)
    print("Conversion successful.")
except Exception as e:
    print(f"Error converting data: {e}")
    raise

# Estimate PSD from the data
print("Estimating PSD from data...")
try:
    psd = h1_data.psd(PSD_SEGMENT_LENGTH * h1_data.sample_rate, avg_method='median')
    psd = interpolate(psd, h1_data.delta_f)
    psd = inverse_spectrum_truncation(psd, int(PSD_SEGMENT_LENGTH * h1_data.sample_rate))
    print("PSD estimation complete.")
except Exception as e:
    print(f"Error estimating PSD: {e}")
    raise

# Generate template bank
print("Generating template bank parameters...")
bank_params = TEMPLATE_BANK_PARAMS.copy()
bank_params["sample-rate"] = int(h1_data.sample_rate)
bank_params["delta-f"] = h1_data.delta_f

try:
    print("Generating template bank...")
    # get_bank returns a numpy structured array of template parameters
    bank = get_bank(**bank_params)
    print(f"Template bank generated with {len(bank)} templates.")
except Exception as e:
    print(f"Error generating template bank: {e}")
    raise

# Matched filtering
print("Starting matched filtering...")
max_snr = 0
best_template = None
best_snr_series = None

for i, template in enumerate(bank):
    try:
        # get_bank returns a numpy structured array, so access fields accordingly
        hp, _ = get_fd_waveform(
            approximant="IMRPhenomPv2",
            mass1=template['mass1'],
            mass2=template['mass2'],
            spin1z=template['spin1z'],
            spin2z=template['spin2z'],
            f_lower=30.0,
            delta_f=h1_data.delta_f,
            distance=1000  # Mpc, arbitrary for SNR ranking
        )
        # Resize waveform to match data length
        if len(hp) > len(h1_data):
            hp = hp[:len(h1_data)]
        elif len(hp) < len(h1_data):
            hp = hp.append_zeros(len(h1_data) - len(hp))
        # Matched filter
        snr = matched_filter(hp, h1_data, psd=psd, low_frequency_cutoff=30.0)
        peak_snr = abs(snr).numpy().max()
        if peak_snr > max_snr:
            max_snr = peak_snr
            best_template = template
            best_snr_series = snr
        if i % 10 == 0 or i == len(bank) - 1:
            print(f"Processed {i+1}/{len(bank)} templates. Current max SNR: {max_snr:.2f}")
    except Exception as e:
        print(f"Error processing template {i}: {e}")

if best_template is None or best_snr_series is None:
    raise RuntimeError("Matched filtering failed to find a best-fit template.")

print("Matched filtering complete.")
print(f"Best-fit template: {best_template}")
print(f"Maximum SNR: {max_snr:.2f}")

# Save matched filtering results
with open(os.path.join(RESULTS_DIR, "best_fit_template.pkl"), "wb") as f:
    pickle.dump(dict(zip(bank.dtype.names, best_template)), f)
best_snr_series.save(os.path.join(RESULTS_DIR, "best_fit_snr_series.hdf"))

# =========================
# Task 4: Parameter Extraction
# =========================
print("\n=== Task 4: Extracting Physical Parameters from Best-fit Template ===")

try:
    # Extract component masses and spins
    m1 = best_template['mass1']
    m2 = best_template['mass2']
    spin1z = best_template['spin1z']
    spin2z = best_template['spin2z']
    distance = best_template['distance'] if 'distance' in best_template.dtype.names else 1000.0  # Mpc

    # Compute final black hole mass and spin
    mf, af = final_mass_spin(m1, m2, spin1z, spin2z)

    # Find merger time (time of maximum SNR)
    snr_abs = abs(best_snr_series)
    max_idx = snr_abs.numpy().argmax()
    merger_time = best_snr_series.sample_times[max_idx]

    # Report parameters
    print(f"Component masses: m1 = {m1:.2f} Msun, m2 = {m2:.2f} Msun")
    print(f"Component spins: spin1z = {spin1z:.2f}, spin2z = {spin2z:.2f}")
    print(f"Final black hole mass: {mf:.2f} Msun")
    print(f"Final black hole spin: {af:.2f}")
    print(f"Luminosity distance: {distance:.2f} Mpc")
    print(f"Merger time (GPS): {merger_time:.6f}")

    # Save extracted parameters
    extracted_parameters = {
        "mass1": float(m1),
        "mass2": float(m2),
        "spin1z": float(spin1z),
        "spin2z": float(spin2z),
        "final_mass": float(mf),
        "final_spin": float(af),
        "distance": float(distance),
        "merger_time": float(merger_time)
    }
    with open(os.path.join(RESULTS_DIR, "extracted_parameters.pkl"), "wb") as f:
        pickle.dump(extracted_parameters, f)
except Exception as e:
    print(f"Error extracting parameters: {e}")
    extracted_parameters = None

print("\n=== Analysis Complete ===")
if extracted_parameters is not None:
    print("Extracted parameters saved to:", os.path.join(RESULTS_DIR, "extracted_parameters.pkl"))
else:
    print("Parameter extraction failed.")