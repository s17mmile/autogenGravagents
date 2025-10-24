# =========================
# GW170608 Matched Filtering Analysis Pipeline
# =========================

# --- Imports ---
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import matched_filter
import h5py

# =========================
# 1. Data Loading & Preprocessing
# =========================

print("\n========== 1. Data Loading & Preprocessing ==========")

# Parameters for GW170608
event_gps = 1180922494.5
duration = 64  # seconds
half_duration = duration / 2
ifo_list = ['H1', 'L1']
bandpass_low = 30
bandpass_high = 300

# Dictionary to store processed data
processed_strain = {}

for ifo in ifo_list:
    print(f"\n--- Processing {ifo} ---")
    try:
        print(f"Fetching {duration}s of open data for {ifo} centered at GPS {event_gps}...")
        strain = TimeSeries.fetch_open_data(
            ifo, event_gps - half_duration, event_gps + half_duration, cache=True
        )
        print(f"Data fetched for {ifo}.")
    except Exception as e:
        print(f"Error fetching data for {ifo}: {e}")
        processed_strain[ifo] = None
        continue

    try:
        print(f"Applying {bandpass_low}-{bandpass_high} Hz bandpass filter to {ifo}...")
        strain_bp = strain.bandpass(bandpass_low, bandpass_high)
        print(f"Bandpass filtering complete for {ifo}.")
    except Exception as e:
        print(f"Error during bandpass filtering for {ifo}: {e}")
        processed_strain[ifo] = None
        continue

    try:
        print(f"Whitening {ifo} strain data...")
        strain_whitened = strain_bp.whiten()
        print(f"Whitening complete for {ifo}.")
    except Exception as e:
        print(f"Error during whitening for {ifo}: {e}")
        processed_strain[ifo] = None
        continue

    processed_strain[ifo] = strain_whitened
    print(f"{ifo} processing complete. Data stored in processed_strain['{ifo}'].")

# Check that both detectors have valid data
if any(processed_strain[ifo] is None for ifo in ifo_list):
    print("\nERROR: Failed to preprocess data for one or more detectors. Exiting pipeline.")
    exit(1)

# =========================
# 2. Template Bank Generation
# =========================

print("\n========== 2. Template Bank Generation ==========")

# Template bank parameters
mass1_vals = np.arange(10, 15, 1)  # 10, 11, 12, 13, 14
mass2_vals = np.arange(7, 12, 1)   # 7, 8, 9, 10, 11
spin_vals = [-0.3, 0.0, 0.3]
sample_rate = 4096
duration_template = 16
f_lower = 30
f_upper = 300
approximant = "IMRPhenomPv2"

# Prepare to store templates and parameters
template_bank = []
template_params = []

total_templates = sum(
    1 for m1 in mass1_vals for m2 in mass2_vals if m2 <= m1 for s1z in spin_vals for s2z in spin_vals
)
template_count = 0
failures_generation = 0

print(f"Generating template bank with {total_templates} templates...")

for m1 in mass1_vals:
    for m2 in mass2_vals:
        if m2 > m1:
            continue  # Enforce m1 >= m2
        for s1z in spin_vals:
            for s2z in spin_vals:
                params = {
                    'mass1': float(m1),
                    'mass2': float(m2),
                    'spin1z': float(s1z),
                    'spin2z': float(s2z),
                    'approximant': approximant,
                    'delta_t': 1.0 / sample_rate,
                    'f_lower': f_lower,
                    'f_final': f_upper,
                    'duration': duration_template
                }
                try:
                    hp, hc = get_td_waveform(
                        approximant=approximant,
                        mass1=m1,
                        mass2=m2,
                        spin1z=s1z,
                        spin2z=s2z,
                        delta_t=1.0 / sample_rate,
                        f_lower=f_lower,
                        f_final=f_upper,
                        duration=duration_template
                    )
                    # Pad or truncate to exact length
                    n_samples = int(duration_template * sample_rate)
                    hp = hp.resize(n_samples)
                    # Store only the 'plus' polarization for the bank
                    template_bank.append(hp.numpy())
                    template_params.append(params)
                    template_count += 1
                    print(f"Template {template_count}/{total_templates} generated: m1={m1}, m2={m2}, s1z={s1z}, s2z={s2z}")
                except Exception as e:
                    print(f"Failed to generate template for m1={m1}, m2={m2}, s1z={s1z}, s2z={s2z}: {e}")
                    failures_generation += 1

print(f"Template bank generation complete. {template_count} templates successfully generated, {failures_generation} failures.")

# Save templates and parameters to an HDF5 file for later use
try:
    with h5py.File("reduced_template_bank.hdf5", "w") as f:
        f.create_dataset("templates", data=np.array(template_bank))
        # Save parameters as a string array for simplicity
        param_strings = np.array([str(p) for p in template_params], dtype='S')
        f.create_dataset("params", data=param_strings)
    print("Template bank saved to 'reduced_template_bank.hdf5'.")
except Exception as e:
    print(f"Error saving template bank: {e}")

if len(template_bank) == 0:
    print("\nERROR: No templates generated. Exiting pipeline.")
    exit(1)

# =========================
# 3. Matched Filtering Analysis
# =========================

print("\n========== 3. Matched Filtering Analysis ==========")

# Convert GWpy TimeSeries to PyCBC TimeSeries for both detectors
pycbc_data = {}
for ifo in ['H1', 'L1']:
    try:
        print(f"Converting {ifo} data to PyCBC TimeSeries...")
        strain = processed_strain[ifo]
        pycbc_data[ifo] = PyCBC_TimeSeries(strain.value, delta_t=1.0/strain.sample_rate.value, epoch=strain.t0.value)
        print(f"{ifo} conversion successful.")
    except Exception as e:
        print(f"Error converting {ifo} data: {e}")
        pycbc_data[ifo] = None

if any(pycbc_data[ifo] is None for ifo in ['H1', 'L1']):
    print("\nERROR: Failed to convert data for one or more detectors. Exiting pipeline.")
    exit(1)

# Prepare to store results
peak_network_snr = -np.inf
peak_template_idx = -1
peak_time = None
snr_series_H1 = []
snr_series_L1 = []
network_snr_series = []
failures_filtering = 0

print(f"Starting matched filtering for {len(template_bank)} templates...")

for idx, (template, params) in enumerate(zip(template_bank, template_params)):
    try:
        # Convert template to PyCBC TimeSeries, align duration/sample rate
        delta_t = pycbc_data['H1'].delta_t
        epoch = pycbc_data['H1'].start_time
        template_ts = PyCBC_TimeSeries(template, delta_t=delta_t, epoch=epoch)
        
        # Matched filter for H1
        snr_H1 = matched_filter(template_ts, pycbc_data['H1'], low_frequency_cutoff=30)
        # Matched filter for L1
        snr_L1 = matched_filter(template_ts, pycbc_data['L1'], low_frequency_cutoff=30)
        
        # Align SNR time series (PyCBC returns SNRs with .sample_times)
        t0 = max(snr_H1.sample_times[0], snr_L1.sample_times[0])
        t1 = min(snr_H1.sample_times[-1], snr_L1.sample_times[-1])
        mask_H1 = (snr_H1.sample_times >= t0) & (snr_H1.sample_times <= t1)
        mask_L1 = (snr_L1.sample_times >= t0) & (snr_L1.sample_times <= t1)
        snr_H1_trim = snr_H1[mask_H1]
        snr_L1_trim = snr_L1[mask_L1]
        
        # Compute network SNR time series
        network_snr = np.sqrt(np.abs(snr_H1_trim) ** 2 + np.abs(snr_L1_trim) ** 2)
        network_snr_series.append(network_snr)
        snr_series_H1.append(snr_H1_trim)
        snr_series_L1.append(snr_L1_trim)
        
        # Find peak network SNR for this template
        max_idx = np.argmax(network_snr)
        max_snr = network_snr[max_idx]
        max_time = snr_H1_trim.sample_times[max_idx]
        
        if max_snr > peak_network_snr:
            peak_network_snr = max_snr
            peak_template_idx = idx
            peak_time = max_time
            print(f"New peak network SNR: {peak_network_snr:.2f} at time {peak_time} (template {idx})")
    except Exception as e:
        print(f"Error in matched filtering for template {idx}: {e}")
        snr_series_H1.append(None)
        snr_series_L1.append(None)
        network_snr_series.append(None)
        failures_filtering += 1
        continue

if peak_template_idx >= 0:
    best_params = template_params[peak_template_idx]
    m1 = best_params['mass1']
    m2 = best_params['mass2']
    chirp_mass = ((m1 * m2) ** (3/5)) / ((m1 + m2) ** (1/5))
    mass_ratio = m1 / m2
    print("\nBest-fit template parameters:")
    print(f"  mass1 = {m1} Msun")
    print(f"  mass2 = {m2} Msun")
    print(f"  spin1z = {best_params['spin1z']}")
    print(f"  spin2z = {best_params['spin2z']}")
    print(f"  chirp mass = {chirp_mass:.4f} Msun")
    print(f"  mass ratio (m1/m2) = {mass_ratio:.4f}")
    print(f"  Peak network SNR = {peak_network_snr:.2f} at time {peak_time}")
else:
    print("No valid template produced a network SNR.")

# Save results for downstream use
matched_filter_results = {
    'snr_series_H1': snr_series_H1,
    'snr_series_L1': snr_series_L1,
    'network_snr_series': network_snr_series,
    'peak_network_snr': peak_network_snr,
    'peak_template_idx': peak_template_idx,
    'peak_time': peak_time,
    'best_params': best_params if peak_template_idx >= 0 else None,
    'chirp_mass': chirp_mass if peak_template_idx >= 0 else None,
    'mass_ratio': mass_ratio if peak_template_idx >= 0 else None
}

# =========================
# 4. Results Comparison & Reporting
# =========================

print("\n========== 4. Results Comparison & Reporting ==========")

# --- Published GW170608 parameters (from GWTC-1 catalog) ---
published = {
    "chirp_mass": 7.9,        # Msun (median, source frame)
    "mass1": 12.0,            # Msun (median, source frame)
    "mass2": 7.0,             # Msun (median, source frame)
    "mass_ratio": 12.0/7.0,   # ~1.71
    "network_snr": 15.0       # Approximate published value
}

# --- Gather statistics from previous steps ---
num_templates = len(template_params)
num_failures = failures_generation + failures_filtering
num_success = num_templates - failures_filtering

peak_network_snr = matched_filter_results['peak_network_snr']
peak_template_idx = matched_filter_results['peak_template_idx']
best_params = matched_filter_results['best_params']
chirp_mass = matched_filter_results['chirp_mass']
mass_ratio = matched_filter_results['mass_ratio']

print("\n===== GW170608 Matched Filtering Results Summary =====\n")

print("Template Bank Statistics:")
print(f"  Total templates attempted: {num_templates}")
print(f"  Templates successfully filtered: {num_success}")
print(f"  Templates failed (errors): {num_failures}")

if best_params is not None:
    print("\nBest-fit Template Parameters (from analysis):")
    print(f"  mass1      = {best_params['mass1']} Msun")
    print(f"  mass2      = {best_params['mass2']} Msun")
    print(f"  spin1z     = {best_params['spin1z']}")
    print(f"  spin2z     = {best_params['spin2z']}")
    print(f"  chirp mass = {chirp_mass:.4f} Msun")
    print(f"  mass ratio = {mass_ratio:.4f}")
    print(f"  Peak network SNR = {peak_network_snr:.2f}")
else:
    print("\nNo valid best-fit template found.")

print("\nPublished GW170608 Parameters (GWTC-1):")
print(f"  mass1      = {published['mass1']} Msun")
print(f"  mass2      = {published['mass2']} Msun")
print(f"  chirp mass = {published['chirp_mass']} Msun")
print(f"  mass ratio = {published['mass_ratio']:.4f}")
print(f"  Network SNR (approx) = {published['network_snr']}")

if best_params is not None:
    print("\nDiscrepancies (Analysis - Published):")
    print(f"  Δ mass1      = {best_params['mass1'] - published['mass1']:.2f} Msun")
    print(f"  Δ mass2      = {best_params['mass2'] - published['mass2']:.2f} Msun")
    print(f"  Δ chirp mass = {chirp_mass - published['chirp_mass']:.2f} Msun")
    print(f"  Δ mass ratio = {mass_ratio - published['mass_ratio']:.2f}")
    print(f"  Δ network SNR= {peak_network_snr - published['network_snr']:.2f}")

print("\nError Handling Summary:")
if num_failures == 0:
    print("  All templates processed successfully.")
else:
    print(f"  {num_failures} templates failed during matched filtering or generation. See logs above for details.")

print("\n===== End of Report =====\n")