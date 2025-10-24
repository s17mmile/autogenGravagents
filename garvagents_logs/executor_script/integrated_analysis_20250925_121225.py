# --- Imports ---
from gwpy.timeseries import TimeSeries
import numpy as np
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.psd import welch
from pycbc.filter import matched_filter
import time

# --- Step 1: Data Download and Preprocessing ---
print("="*60)
print("Step 1: Downloading and preprocessing LIGO H1 and L1 strain data for GW170608")

gw170608_gps = 1180922494.5
segment_duration = 64  # seconds
start = gw170608_gps - segment_duration / 2
end = gw170608_gps + segment_duration / 2

detectors = ['H1', 'L1']
h1_strain_processed = None
l1_strain_processed = None

for det in detectors:
    try:
        print(f"  Fetching {det} data from {start} to {end} (GPS)...")
        strain = TimeSeries.fetch_open_data(det, start, end, cache=True)
        print(f"    Data fetched for {det}. Applying bandpass filter (30-300 Hz)...")
        strain_bp = strain.bandpass(30, 300)
        print(f"    Bandpass applied for {det}. Whitening...")
        strain_whitened = strain_bp.whiten()
        print(f"    Whitening complete for {det}.")
        if det == 'H1':
            h1_strain_processed = strain_whitened
        elif det == 'L1':
            l1_strain_processed = strain_whitened
    except Exception as e:
        print(f"  ERROR: Failed to process {det} data: {e}")

if h1_strain_processed is not None and l1_strain_processed is not None:
    print("Data download and preprocessing complete for both H1 and L1.")
else:
    print("ERROR: One or both detectors failed to process. Exiting.")
    exit(1)

# --- Step 2: Template Bank Generation ---
print("="*60)
print("Step 2: Generating template bank (IMRPhenomPv2, mass/spin grid)")

m1_vals = np.arange(8.0, 15.0 + 0.01, 0.5)
m2_vals = np.arange(6.0, 12.0 + 0.01, 0.5)
spin_vals = [-0.5, 0.0, 0.5]
f_lower = 30.0
f_upper = 300.0  # Not directly used in get_td_waveform, but for reference
duration = 64
sample_rate = 4096
delta_t = 1.0 / sample_rate
approximant = "IMRPhenomPv2"

template_bank = []
failures = []

count = 0
for m1 in m1_vals:
    for m2 in m2_vals:
        if m2 > m1:
            continue  # Only allow m1 >= m2
        for spin1z in spin_vals:
            for spin2z in spin_vals:
                params = {
                    'approximant': approximant,
                    'mass1': float(m1),
                    'mass2': float(m2),
                    'spin1z': float(spin1z),
                    'spin2z': float(spin2z),
                    'delta_t': delta_t,
                    'f_lower': f_lower,
                    'duration': duration
                }
                try:
                    hp, hc = get_td_waveform(
                        approximant=approximant,
                        mass1=m1,
                        mass2=m2,
                        spin1z=spin1z,
                        spin2z=spin2z,
                        delta_t=delta_t,
                        f_lower=f_lower,
                        duration=duration
                    )
                    # Store only parameters for identification (waveform can be regenerated as needed)
                    template_bank.append({'params': params})
                    count += 1
                    if count % 100 == 0:
                        print(f"  Generated {count} templates...")
                except Exception as e:
                    failures.append({'params': params, 'error': str(e)})
                    print(f"  ERROR: Failed to generate template for {params}: {e}")

print(f"Template bank generation complete: {len(template_bank)} templates generated, {len(failures)} failures.")

if not template_bank:
    print("ERROR: No templates generated. Exiting.")
    exit(1)

# --- Step 3: Matched Filtering and Network SNR ---
print("="*60)
print("Step 3: Matched filtering and network SNR computation")

def gwpy_to_pycbc(ts):
    # Handles both astropy and float dt/t0
    dt = ts.dt.value if hasattr(ts.dt, 'value') else ts.dt
    t0 = ts.t0.value if hasattr(ts.t0, 'value') else ts.t0
    return PyCBC_TimeSeries(ts.value, delta_t=dt, epoch=t0)

try:
    print("Converting GWPy TimeSeries to PyCBC TimeSeries...")
    h1_pycbc = gwpy_to_pycbc(h1_strain_processed)
    l1_pycbc = gwpy_to_pycbc(l1_strain_processed)
except Exception as e:
    print(f"ERROR: Could not convert GWPy to PyCBC TimeSeries: {e}")
    exit(1)

def estimate_psd(strain, seglen=4, avg_method='median'):
    # Try several segment lengths for robust PSD estimation
    total_samples = len(strain)
    dt = strain.delta_t
    seglens_to_try = [4, 8, 16, 2]  # Try 4s, 8s, 16s, 2s
    for seglen_try in seglens_to_try:
        seg_len_samples = int(seglen_try / dt)
        nseg = total_samples // seg_len_samples
        if seg_len_samples < 2 or nseg < 2:
            continue  # Not enough data for this segment length
        try:
            print(f"  Estimating PSD (segment length {seglen_try}s, {seg_len_samples} samples, {nseg} segments, method {avg_method})...")
            psd = welch(strain, seg_len=seg_len_samples, avg_method=avg_method)
            return psd
        except ValueError as e:
            print(f"  WARNING: PSD estimation failed for seglen={seglen_try}s: {e}")
            continue
    # If all fail, raise error
    raise RuntimeError("PSD estimation failed for all tried segment lengths.")

print("Estimating PSDs for H1 and L1...")
try:
    h1_psd = estimate_psd(h1_pycbc)
    l1_psd = estimate_psd(l1_pycbc)
except Exception as e:
    print(f"ERROR: PSD estimation failed: {e}")
    exit(1)

snr_results = []
max_network_snr = -np.inf
best_fit_params = None

print("Starting matched filtering for all templates...")
for idx, template in enumerate(template_bank):
    params = template['params']
    try:
        # Generate template waveform
        hp, _ = get_td_waveform(
            approximant=params['approximant'],
            mass1=params['mass1'],
            mass2=params['mass2'],
            spin1z=params['spin1z'],
            spin2z=params['spin2z'],
            delta_t=params['delta_t'],
            f_lower=params['f_lower'],
            duration=params['duration']
        )
        hp.resize(len(h1_pycbc))
        # Matched filter for H1
        snr_h1 = matched_filter(hp, h1_pycbc, psd=h1_psd, low_frequency_cutoff=params['f_lower'])
        peak_snr_h1 = abs(snr_h1).numpy().max()
        # Matched filter for L1
        snr_l1 = matched_filter(hp, l1_pycbc, psd=l1_psd, low_frequency_cutoff=params['f_lower'])
        peak_snr_l1 = abs(snr_l1).numpy().max()
        # Network SNR
        network_snr = np.sqrt(peak_snr_h1**2 + peak_snr_l1**2)
        # Store results
        snr_results.append({
            'params': params,
            'peak_snr_h1': peak_snr_h1,
            'peak_snr_l1': peak_snr_l1,
            'network_snr': network_snr
        })
        # Track best-fit template
        if network_snr > max_network_snr:
            max_network_snr = network_snr
            best_fit_params = snr_results[-1]
        if (idx+1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(template_bank)} templates...")
    except Exception as e:
        print(f"  ERROR: Matched filtering failed for template {params}: {e}")

print(f"Matched filtering complete. {len(snr_results)} templates processed.")
if best_fit_params is not None:
    print(f"Peak network SNR: {best_fit_params['network_snr']:.2f} (template: {best_fit_params['params']})")
else:
    print("ERROR: No valid matched filter results found. Exiting.")
    exit(1)

# --- Step 4: Parameter Estimation and Reporting ---
print("="*60)
print("Step 4: Parameter estimation and reporting")

def chirp_mass(m1, m2):
    return ((m1 * m2) ** (3/5)) / ((m1 + m2) ** (1/5))

def mass_ratio(m1, m2):
    return m2 / m1

def total_mass(m1, m2):
    return m1 + m2

published_m1 = 12.0
published_m2 = 7.0

try:
    print("Extracting best-fit parameters from peak SNR template...")
    best_params = best_fit_params['params']
    m1 = best_params['mass1']
    m2 = best_params['mass2']
    spin1z = best_params['spin1z']
    spin2z = best_params['spin2z']
    peak_network_snr = best_fit_params['network_snr']
    peak_snr_h1 = best_fit_params['peak_snr_h1']
    peak_snr_l1 = best_fit_params['peak_snr_l1']

    # Derived parameters
    mc = chirp_mass(m1, m2)
    mtot = total_mass(m1, m2)
    q = mass_ratio(m1, m2)

    # Find time offset of peak SNR (for H1 and L1)
    hp, _ = get_td_waveform(
        approximant=best_params['approximant'],
        mass1=m1,
        mass2=m2,
        spin1z=spin1z,
        spin2z=spin2z,
        delta_t=best_params['delta_t'],
        f_lower=best_params['f_lower'],
        duration=best_params['duration']
    )
    hp.resize(len(h1_pycbc))
    snr_h1_series = matched_filter(hp, h1_pycbc, psd=h1_psd, low_frequency_cutoff=best_params['f_lower'])
    snr_l1_series = matched_filter(hp, l1_pycbc, psd=l1_psd, low_frequency_cutoff=best_params['f_lower'])
    peak_idx_h1 = np.argmax(abs(snr_h1_series))
    peak_idx_l1 = np.argmax(abs(snr_l1_series))
    time_offset_h1 = snr_h1_series.sample_times[peak_idx_h1]
    time_offset_l1 = snr_l1_series.sample_times[peak_idx_l1]

    # Estimate uncertainties: use templates within 90% of peak network SNR
    snrs = np.array([r['network_snr'] for r in snr_results])
    threshold = 0.9 * peak_network_snr
    close_idxs = np.where(snrs >= threshold)[0]
    close_templates = [snr_results[i] for i in close_idxs]
    m1s = np.array([t['params']['mass1'] for t in close_templates])
    m2s = np.array([t['params']['mass2'] for t in close_templates])
    mcs = chirp_mass(m1s, m2s)
    mtots = m1s + m2s
    qs = m2s / m1s

    def estimate_range(arr):
        return np.min(arr), np.max(arr), np.mean(arr), np.std(arr)

    mc_min, mc_max, mc_mean, mc_std = estimate_range(mcs)
    mtot_min, mtot_max, mtot_mean, mtot_std = estimate_range(mtots)
    q_min, q_max, q_mean, q_std = estimate_range(qs)

    print("\n===== GW170608 Parameter Estimation Report =====")
    print(f"Template bank size: {len(template_bank)}")
    print(f"Peak network SNR: {peak_network_snr:.2f} (H1: {peak_snr_h1:.2f}, L1: {peak_snr_l1:.2f})")
    print(f"Best-fit parameters:")
    print(f"  m1 = {m1:.2f} M☉, m2 = {m2:.2f} M☉, spin1z = {spin1z:.2f}, spin2z = {spin2z:.2f}")
    print(f"  Chirp mass: {mc:.2f} M☉")
    print(f"  Total mass: {mtot:.2f} M☉")
    print(f"  Mass ratio: {q:.2f}")
    print(f"  Time offset (H1): {time_offset_h1:.4f} s, (L1): {time_offset_l1:.4f} s")
    print("\nParameter uncertainties (range among templates with network SNR ≥ 90% of peak):")
    print(f"  Chirp mass: {mc_min:.2f} – {mc_max:.2f} M☉ (mean: {mc_mean:.2f}, std: {mc_std:.2f})")
    print(f"  Total mass: {mtot_min:.2f} – {mtot_max:.2f} M☉ (mean: {mtot_mean:.2f}, std: {mtot_std:.2f})")
    print(f"  Mass ratio: {q_min:.2f} – {q_max:.2f} (mean: {q_mean:.2f}, std: {q_std:.2f})")
    print("\nComparison to published GW170608 values:")
    print(f"  Published m1 ≈ {published_m1:.1f} M☉, m2 ≈ {published_m2:.1f} M☉")
    print(f"  Δm1 = {m1 - published_m1:.2f} M☉, Δm2 = {m2 - published_m2:.2f} M☉")
    print("===============================================")
except Exception as e:
    print(f"ERROR: Parameter estimation/reporting failed: {e}")

print("\nWorkflow complete.")