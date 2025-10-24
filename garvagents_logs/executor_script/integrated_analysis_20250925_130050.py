#!/usr/bin/env python3
# GW170608 Matched Filtering and Parameter Estimation Pipeline

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
gps_time = 1180922494.5
duration = 64  # seconds
start = gps_time - duration / 2
end = gps_time + duration / 2
detectors = ['H1', 'L1']
sample_rate = 4096
delta_t = 1.0 / sample_rate
f_lower = 30
f_upper = 300

# Template bank parameters
mass1_vals = np.arange(8, 15.0 + 0.01, 0.5)   # 8 to 15 M☉, inclusive
mass2_vals = np.arange(6, 12.0 + 0.01, 0.5)   # 6 to 12 M☉, inclusive
spin_vals = [-0.5, 0.0, 0.5]
approximant = "IMRPhenomPv2"

# Published GW170608 parameters for comparison
published_m1 = 12.0
published_m2 = 7.0

# =========================
# 1. DATA LOADING & PREPROCESSING
# =========================
print("\n========== 1. DATA LOADING & PREPROCESSING ==========")
strain_data = {}
strain_data_bandpassed = {}
strain_data_whitened = {}

for det in detectors:
    print(f"\nProcessing {det} data...")
    try:
        print(f"Fetching open data for {det} from {start} to {end}...")
        ts = TimeSeries.fetch_open_data(det, start, end, cache=True)
        strain_data[det] = ts
        print(f"Data fetched for {det}.")
        
        print(f"Applying 30-300 Hz bandpass filter to {det} data...")
        ts_bp = ts.bandpass(f_lower, f_upper)
        strain_data_bandpassed[det] = ts_bp
        print(f"Bandpass filtering complete for {det}.")
        
        print(f"Whitening {det} data...")
        ts_white = ts_bp.whiten()
        strain_data_whitened[det] = ts_white
        print(f"Whitening complete for {det}.")
        
    except Exception as e:
        print(f"Error processing {det}: {e}")
        sys.exit(1)

print("\nData loading and preprocessing complete.")

# =========================
# 2. TEMPLATE BANK GENERATION
# =========================
print("\n========== 2. TEMPLATE BANK GENERATION ==========")
template_bank = {}
total_templates = 0
for m1 in mass1_vals:
    for m2 in mass2_vals:
        if m2 > m1:
            continue  # Enforce m1 >= m2
        for s1z in spin_vals:
            for s2z in spin_vals:
                key = (round(m1,2), round(m2,2), s1z, s2z)
                try:
                    hp, hc = get_td_waveform(
                        approximant=approximant,
                        mass1=m1, mass2=m2,
                        spin1z=s1z, spin2z=s2z,
                        delta_t=delta_t,
                        f_lower=f_lower,
                        f_final=f_upper,
                        duration=duration
                    )
                    # Truncate or pad waveform to exactly 64s
                    target_len = int(duration * sample_rate)
                    hp = hp.crop(0, duration)
                    if len(hp) < target_len:
                        pad = np.zeros(target_len - len(hp))
                        hp_data = np.concatenate([hp.numpy(), pad])
                    else:
                        hp_data = hp.numpy()[:target_len]
                    template_bank[key] = hp_data
                    total_templates += 1
                    if total_templates % 100 == 0:
                        print(f"Generated {total_templates} templates...")
                except Exception as e:
                    print(f"Error generating template for m1={m1}, m2={m2}, s1z={s1z}, s2z={s2z}: {e}", file=sys.stderr)
print(f"Template bank generation complete. {total_templates} templates generated.")

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
        print(f"Converted {det} data to PyCBC TimeSeries.")
    except Exception as e:
        print(f"Error converting {det} data: {e}", file=sys.stderr)
        sys.exit(1)

max_network_snr = 0
max_template = None
max_time = None
max_snr_series = None

template_idx = 0
for key, hp_data in template_bank.items():
    template_idx += 1
    try:
        template = PyCBC_TimeSeries(hp_data, delta_t=delta_t)
        snr_series = {}
        for det in detectors:
            snr = matched_filter(template, pycbc_data[det], low_frequency_cutoff=f_lower)
            snr_series[det] = snr
        min_len = min(len(snr_series['H1']), len(snr_series['L1']))
        snr_h1 = snr_series['H1'][:min_len]
        snr_l1 = snr_series['L1'][:min_len]
        network_snr = np.sqrt(np.abs(snr_h1)**2 + np.abs(snr_l1)**2)
        peak_idx = np.argmax(network_snr)
        peak_snr = network_snr[peak_idx]
        peak_time = snr_h1.sample_times[peak_idx]
        if peak_snr > max_network_snr:
            max_network_snr = peak_snr
            max_template = key
            max_time = peak_time
            max_snr_series = {'H1': snr_h1, 'L1': snr_l1, 'network': network_snr}
        if template_idx % 100 == 0 or template_idx == total_templates:
            print(f"Processed {template_idx}/{total_templates} templates. Current max network SNR: {max_network_snr:.2f}")
    except Exception as e:
        print(f"Error in matched filtering for template {key}: {e}", file=sys.stderr)

print("\nMatched filtering complete.")
print(f"Maximum network SNR: {max_network_snr:.2f}")
print(f"Best template (m1, m2, s1z, s2z): {max_template}")
print(f"Peak time (GPS): {max_time}")

# =========================
# 4. PARAMETER ESTIMATION & REPORTING
# =========================
print("\n========== 4. PARAMETER ESTIMATION & REPORTING ==========")
best_m1, best_m2, best_s1z, best_s2z = max_template
best_total_mass = best_m1 + best_m2
best_mass_ratio = best_m2 / best_m1
best_chirp_mass = ((best_m1 * best_m2) ** (3/5)) / (best_total_mass ** (1/5))
best_time_offset = max_time  # GPS time of peak SNR

print("\nBest-fit parameters from peak SNR template:")
print(f"  m1 = {best_m1:.2f} M☉")
print(f"  m2 = {best_m2:.2f} M☉")
print(f"  spin1z = {best_s1z:.2f}")
print(f"  spin2z = {best_s2z:.2f}")
print(f"  Chirp mass = {best_chirp_mass:.2f} M☉")
print(f"  Total mass = {best_total_mass:.2f} M☉")
print(f"  Mass ratio (q = m2/m1) = {best_mass_ratio:.3f}")
print(f"  Time offset (GPS) = {best_time_offset:.6f}")
print(f"  Peak network SNR = {max_network_snr:.2f}")

# Estimate uncertainties by examining SNRs of neighboring templates
def get_neighbors(param_grid, center, step):
    m1, m2, s1z, s2z = center
    neighbors = []
    for dm1 in [-step, 0, step]:
        for dm2 in [-step, 0, step]:
            for ds1z in [-1, 0, 1]:
                for ds2z in [-1, 0, 1]:
                    nm1 = round(m1 + dm1, 2)
                    nm2 = round(m2 + dm2, 2)
                    ns1z = s1z + ds1z * 0.5
                    ns2z = s2z + ds2z * 0.5
                    if (nm1, nm2, ns1z, ns2z) in param_grid and nm2 <= nm1:
                        neighbors.append((nm1, nm2, ns1z, ns2z))
    return neighbors

print("\nEstimating uncertainties from local SNR neighborhood...")
param_grid = set(template_bank.keys())
neighbor_keys = get_neighbors(param_grid, max_template, 0.5)
neighbor_snrs = []
for key in neighbor_keys:
    try:
        hp_data = template_bank[key]
        template = PyCBC_TimeSeries(hp_data, delta_t=delta_t)
        snr_h1 = matched_filter(template, pycbc_data['H1'], low_frequency_cutoff=f_lower)
        snr_l1 = matched_filter(template, pycbc_data['L1'], low_frequency_cutoff=f_lower)
        min_len = min(len(snr_h1), len(snr_l1))
        snr_h1 = snr_h1[:min_len]
        snr_l1 = snr_l1[:min_len]
        network_snr = np.sqrt(np.abs(snr_h1)**2 + np.abs(snr_l1)**2)
        peak_snr = np.max(network_snr)
        neighbor_snrs.append((key, peak_snr))
    except Exception as e:
        print(f"Error computing SNR for neighbor {key}: {e}")

snr_values = np.array([snr for _, snr in neighbor_snrs])
param_values = np.array([key for key, _ in neighbor_snrs])
if len(snr_values) == 0:
    print("Warning: No neighbor SNRs found for uncertainty estimation.")
    m1_unc = (best_m1, best_m1)
    m2_unc = (best_m2, best_m2)
    chirp_unc = (best_chirp_mass, best_chirp_mass)
else:
    snr_peak = np.max(snr_values)
    snr_1sigma = snr_peak - 1.0
    within_1sigma = param_values[snr_values >= snr_1sigma]
    if len(within_1sigma) > 0:
        m1s = within_1sigma[:,0].astype(float)
        m2s = within_1sigma[:,1].astype(float)
        chirp_masses = ((m1s * m2s) ** (3/5)) / ((m1s + m2s) ** (1/5))
        m1_unc = (np.min(m1s), np.max(m1s))
        m2_unc = (np.min(m2s), np.max(m2s))
        chirp_unc = (np.min(chirp_masses), np.max(chirp_masses))
    else:
        m1_unc = (best_m1, best_m1)
        m2_unc = (best_m2, best_m2)
        chirp_unc = (best_chirp_mass, best_chirp_mass)

print("\nEstimated 1-sigma uncertainties (from local SNR neighborhood):")
print(f"  m1: {m1_unc[0]:.2f} – {m1_unc[1]:.2f} M☉")
print(f"  m2: {m2_unc[0]:.2f} – {m2_unc[1]:.2f} M☉")
print(f"  Chirp mass: {chirp_unc[0]:.2f} – {chirp_unc[1]:.2f} M☉")

print("\nComparison with published GW170608 parameters:")
print(f"  Published m1 ≈ {published_m1:.2f} M☉")
print(f"  Published m2 ≈ {published_m2:.2f} M☉")
print(f"  Published chirp mass ≈ {((published_m1*published_m2)**(3/5))/((published_m1+published_m2)**(1/5)):.2f} M☉")

mass1_vals_bank = sorted(set([k[0] for k in template_bank.keys()]))
mass2_vals_bank = sorted(set([k[1] for k in template_bank.keys()]))
spin_vals_bank = sorted(set([k[2] for k in template_bank.keys()]))
num_templates = len(template_bank)

print("\nTemplate bank coverage:")
print(f"  Mass1 range: {mass1_vals_bank[0]:.2f} – {mass1_vals_bank[-1]:.2f} M☉ ({len(mass1_vals_bank)} values)")
print(f"  Mass2 range: {mass2_vals_bank[0]:.2f} – {mass2_vals_bank[-1]:.2f} M☉ ({len(mass2_vals_bank)} values)")
print(f"  Spin values: {spin_vals_bank} ({len(spin_vals_bank)} values)")
print(f"  Total templates: {num_templates}")

print("\nSNR significance:")
print(f"  Peak network SNR: {max_network_snr:.2f}")
if max_network_snr > 8:
    print("  This is a highly significant detection (SNR > 8).")
elif max_network_snr > 5:
    print("  This is a marginal detection (SNR > 5).")
else:
    print("  This is a low-significance detection (SNR ≤ 5).")

# =========================
# OPTIONAL: SAVE RESULTS
# =========================
try:
    np.save("best_template.npy", np.array(max_template))
    np.save("max_snr_series_network.npy", max_snr_series['network'])
    np.save("max_snr_series_H1.npy", max_snr_series['H1'])
    np.save("max_snr_series_L1.npy", max_snr_series['L1'])
    print("\nSaved best-fit template parameters and SNR time series to disk.")
except Exception as e:
    print(f"Warning: Could not save results to disk: {e}")

print("\n========== PIPELINE COMPLETE ==========")