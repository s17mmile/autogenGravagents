#!/usr/bin/env python3
# GW170608 Reduced Template Bank Matched Filtering Pipeline

# =========================
# IMPORTS AND SETUP
# =========================
import numpy as np
import sys
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import matched_filter

# =========================
# PARAMETERS
# =========================
# Event and data parameters
gps_center = 1180922494.5
duration_data = 64  # seconds for data download
start = gps_center - duration_data/2
end = gps_center + duration_data/2
detectors = ['H1', 'L1']
bandpass = (30, 300)

# Template bank parameters
mass1_vals = np.arange(10, 15, 1.0)  # 10, 11, 12, 13, 14
mass2_vals = np.arange(7, 12, 1.0)   # 7, 8, 9, 10, 11
spin_vals = [-0.3, 0.0, 0.3]
sample_rate = 4096
duration_template = 16  # seconds for templates
f_lower = 30
f_upper = 300

# Published GW170608 values (from LIGO/Virgo GWTC-1 catalog)
published = {
    "chirp_mass": 7.9,  # solar masses
    "chirp_mass_err": 0.2,
    "mass_ratio": 0.67,  # m2/m1
    "mass_ratio_err": 0.07
}

# =========================
# 1. DATA LOADING & PREPROCESSING
# =========================
print("\n========== 1. DATA LOADING & PREPROCESSING ==========")
strain_data_whitened = {}

print(f"Downloading and preprocessing data for GW170608 (GPS {gps_center})...")
for det in detectors:
    try:
        print(f"Fetching {det} data from {start} to {end}...")
        ts = TimeSeries.fetch_open_data(det, start, end, cache=True)
        print(f"{det}: Data fetched. Applying bandpass filter {bandpass[0]}-{bandpass[1]} Hz...")
        ts_bp = ts.bandpass(bandpass[0], bandpass[1])
        print(f"{det}: Bandpass filter applied. Whitening...")
        ts_whiten = ts_bp.whiten()
        print(f"{det}: Whitening complete.")
        strain_data_whitened[det] = ts_whiten
    except Exception as e:
        print(f"Error processing {det}: {e}")
        sys.exit(1)

print("Data loading and preprocessing complete.")

# =========================
# 2. TEMPLATE BANK GENERATION (WITH ERROR HANDLING)
# =========================
print("\n========== 2. TEMPLATE BANK GENERATION ==========")
template_bank = {}
failed_templates = []

print("Generating reduced template bank with IMRPhenomPv2...")
total = len(mass1_vals) * len(mass2_vals) * len(spin_vals) * len(spin_vals)
count = 0

for m1 in mass1_vals:
    for m2 in mass2_vals:
        if m2 > m1:
            continue  # enforce m1 >= m2
        for s1z in spin_vals:
            for s2z in spin_vals:
                count += 1
                key = (float(m1), float(m2), float(s1z), float(s2z))
                try:
                    hp, _ = get_td_waveform(
                        approximant="IMRPhenomPv2",
                        mass1=m1, mass2=m2,
                        spin1z=s1z, spin2z=s2z,
                        delta_t=1.0/sample_rate,
                        f_lower=f_lower,
                        f_final=f_upper,
                        duration=duration_template
                    )
                    # Resample to fixed length (zero-pad or truncate as needed)
                    target_len = int(sample_rate * duration_template)
                    hp = hp.resize(target_len)
                    template_bank[key] = hp.numpy()
                    if count % 20 == 0 or count == total:
                        print(f"Generated {count}/{total} templates...")
                except Exception as e:
                    print(f"Failed to generate template {key}: {e}", file=sys.stderr)
                    failed_templates.append((key, str(e)))

print(f"Template bank generation complete: {len(template_bank)} templates generated, {len(failed_templates)} failures.")

# Save template bank and failures for reproducibility
try:
    np.save("template_bank_keys.npy", np.array(list(template_bank.keys()), dtype=object))
    np.save("failed_templates.npy", np.array(failed_templates, dtype=object))
    print("Saved template bank keys and failed templates to disk.")
except Exception as e:
    print(f"Warning: Could not save template bank info: {e}")

# =========================
# 3. MATCHED FILTERING & SNR CALCULATION
# =========================
print("\n========== 3. MATCHED FILTERING & SNR CALCULATION ==========")
# Convert GWpy TimeSeries to PyCBC TimeSeries for both detectors
pycbc_data = {}
for det in detectors:
    try:
        ts = strain_data_whitened[det]
        pycbc_data[det] = PyCBC_TimeSeries(ts.value, delta_t=ts.dt.value, epoch=ts.t0.value)
        print(f"{det}: Converted GWpy TimeSeries to PyCBC TimeSeries.")
    except Exception as e:
        print(f"Error converting {det} data: {e}")
        sys.exit(1)

max_network_snr = -np.inf
max_template = None
max_time = None

snr_results = {}  # Store SNR time series for each template

print("Starting matched filtering for all templates...")
template_keys = list(template_bank.keys())
for idx, key in enumerate(template_keys):
    try:
        hp = template_bank[key]
        # Create PyCBC TimeSeries for template
        template = PyCBC_TimeSeries(hp, delta_t=1.0/sample_rate)
        snr_h1 = matched_filter(template, pycbc_data['H1'], low_frequency_cutoff=f_lower)
        snr_l1 = matched_filter(template, pycbc_data['L1'], low_frequency_cutoff=f_lower)
        # Align lengths
        min_len = min(len(snr_h1), len(snr_l1))
        snr_h1 = snr_h1[:min_len]
        snr_l1 = snr_l1[:min_len]
        # Compute network SNR time series
        network_snr = np.sqrt(np.abs(snr_h1)**2 + np.abs(snr_l1)**2)
        # Find peak
        peak_idx = np.argmax(network_snr)
        peak_snr = network_snr[peak_idx]
        peak_time = snr_h1.sample_times[peak_idx]
        # Store results
        snr_results[key] = {
            'snr_h1': snr_h1,
            'snr_l1': snr_l1,
            'network_snr': network_snr,
            'peak_snr': peak_snr,
            'peak_time': peak_time
        }
        if peak_snr > max_network_snr:
            max_network_snr = peak_snr
            max_template = key
            max_time = peak_time
        if (idx+1) % 20 == 0 or (idx+1) == len(template_keys):
            print(f"Processed {idx+1}/{len(template_keys)} templates...")
    except Exception as e:
        print(f"Error in matched filtering for template {key}: {e}")

print(f"Matched filtering complete. Peak network SNR: {max_network_snr:.2f} for template {max_template} at time {max_time:.6f}")

# Save SNR results for reproducibility
try:
    np.save("snr_results_keys.npy", np.array(list(snr_results.keys()), dtype=object))
    np.save("max_template.npy", np.array(max_template))
    np.save("max_network_snr.npy", np.array(max_network_snr))
    print("Saved SNR results and best-fit template info to disk.")
except Exception as e:
    print(f"Warning: Could not save SNR results: {e}")

# =========================
# 4. PARAMETER EXTRACTION & COMPARISON
# =========================
print("\n========== 4. PARAMETER EXTRACTION & COMPARISON ==========")

def chirp_mass(m1, m2):
    """Compute chirp mass."""
    return ((m1 * m2) ** (3/5)) / ((m1 + m2) ** (1/5))

def mass_ratio(m1, m2):
    """Compute mass ratio (m2/m1, with m1 >= m2)."""
    return m2 / m1

# Extract best-fit parameters
m1, m2, s1z, s2z = max_template
mc = chirp_mass(m1, m2)
q = mass_ratio(m1, m2)

print("=== Best-fit Template Parameters ===")
print(f"Component masses: m1 = {m1:.2f} M_sun, m2 = {m2:.2f} M_sun")
print(f"Aligned spins: s1z = {s1z:.2f}, s2z = {s2z:.2f}")
print(f"Chirp mass: {mc:.2f} M_sun")
print(f"Mass ratio: {q:.2f}")
print(f"Peak network SNR: {max_network_snr:.2f}")

print("\n=== Published GW170608 Parameters ===")
print(f"Chirp mass: {published['chirp_mass']} ± {published['chirp_mass_err']} M_sun")
print(f"Mass ratio: {published['mass_ratio']} ± {published['mass_ratio_err']}")

# Compare best-fit to published values
mc_diff = abs(mc - published['chirp_mass'])
q_diff = abs(q - published['mass_ratio'])
print("\n=== Comparison ===")
print(f"Chirp mass difference: {mc_diff:.2f} M_sun")
print(f"Mass ratio difference: {q_diff:.2f}")

# Template bank statistics
mass1_vals_bank = sorted(set([k[0] for k in template_bank.keys()]))
mass2_vals_bank = sorted(set([k[1] for k in template_bank.keys()]))
spin_vals_bank = sorted(set([k[2] for k in template_bank.keys()] + [k[3] for k in template_bank.keys()]))

print("\n=== Template Bank Statistics ===")
print(f"Primary mass range: {min(mass1_vals_bank):.1f} - {max(mass1_vals_bank):.1f} M_sun")
print(f"Secondary mass range: {min(mass2_vals_bank):.1f} - {max(mass2_vals_bank):.1f} M_sun")
print(f"Spin values: {spin_vals_bank}")
print(f"Total templates attempted: {len(template_bank) + len(failed_templates)}")
print(f"Templates successfully generated: {len(template_bank)}")
print(f"Templates failed: {len(failed_templates)}")
print(f"Peak network SNR: {max_network_snr:.2f} (template: {max_template})")

print("\n========== PIPELINE COMPLETE ==========")