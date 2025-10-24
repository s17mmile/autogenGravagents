# --- Imports ---
import numpy as np
import itertools
import time
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.psd import welch
from pycbc.filter import matched_filter

# --- Start timer ---
pipeline_start_time = time.time()

# --- Check for preprocessed strain data ---
try:
    h1_strain_processed
    l1_strain_processed
except NameError:
    print("ERROR: Preprocessed strain data (h1_strain_processed, l1_strain_processed) not found in memory.")
    print("Please run the preprocessing workflow first.")
    exit(1)

# --- Task 1: Template Bank Generation ---
m1_vals = np.arange(8, 15.01, 0.5)   # 8 to 15, inclusive, 0.5 steps
m2_vals = np.arange(6, 12.01, 0.5)   # 6 to 12, inclusive, 0.5 steps
spin_vals = [-0.5, 0.0, 0.5]
f_lower = 30
f_upper = 300  # Not directly used in waveform generation, but can be stored
delta_t = 1.0 / 4096
duration = 64

template_bank = []
failed_templates = []

print("="*60)
print("Step 1: Generating template bank for IMRPhenomPv2...")
total_templates = 0

for m1 in m1_vals:
    for m2 in m2_vals:
        if m1 < m2:
            continue  # Only allow m1 >= m2
        for spin1z in spin_vals:
            for spin2z in spin_vals:
                params = {
                    'mass1': m1,
                    'mass2': m2,
                    'spin1z': spin1z,
                    'spin2z': spin2z,
                    'approximant': 'IMRPhenomPv2',
                    'f_lower': f_lower,
                    'delta_t': delta_t
                }
                try:
                    hp, hc = get_td_waveform(**params, duration=duration)
                    template_bank.append({
                        'params': params,
                        'hp': hp,
                        'hc': hc
                    })
                    total_templates += 1
                    if total_templates % 100 == 0:
                        print(f"Generated {total_templates} templates so far...")
                except Exception as e:
                    failed_templates.append({'params': params, 'error': str(e)})
                    print(f"Failed to generate template for params {params}: {e}")

print(f"Template bank generation complete. Total templates: {total_templates}")
if failed_templates:
    print(f"Failed to generate {len(failed_templates)} templates. See error logs for details.")

# --- Task 2: Matched Filtering ---
def gwpy_to_pycbc(ts):
    # Convert GWpy TimeSeries to PyCBC TimeSeries
    return PyCBC_TimeSeries(ts.value, delta_t=ts.dt.value if hasattr(ts.dt, 'value') else ts.dt, epoch=ts.t0.value if hasattr(ts.t0, 'value') else ts.t0)

print("="*60)
print("Step 2: Preparing strain data and estimating PSDs...")

try:
    h1_pycbc = gwpy_to_pycbc(h1_strain_processed)
    l1_pycbc = gwpy_to_pycbc(l1_strain_processed)
except Exception as e:
    print(f"Error converting GWpy to PyCBC TimeSeries: {e}")
    exit(1)

def estimate_psd(strain, seglen=4, avg_method='median'):
    try:
        psd = welch(strain, seg_len=seglen * int(1/strain.delta_t), avg_method=avg_method)
        psd = psd.replace(0, np.inf)
        return psd
    except Exception as e:
        print(f"Error estimating PSD: {e}")
        raise

print("Estimating PSDs for H1 and L1...")
h1_psd = estimate_psd(h1_pycbc)
l1_psd = estimate_psd(l1_pycbc)
print("PSD estimation complete.")

print("="*60)
print("Step 3: Matched filtering for all templates...")
snr_results = []
mf_failed_templates = []

for idx, template in enumerate(template_bank):
    params = template['params']
    hp = template['hp']
    try:
        if hp.delta_t != h1_pycbc.delta_t:
            hp = hp.resample(h1_pycbc.delta_t)
    except Exception as e:
        mf_failed_templates.append({'params': params, 'error': f"Resampling error: {e}"})
        print(f"Template {idx}: Resampling error: {e}")
        continue

    # Matched filter for H1
    try:
        snr_h1 = matched_filter(hp, h1_pycbc, psd=h1_psd, low_frequency_cutoff=params['f_lower'])
    except Exception as e:
        mf_failed_templates.append({'params': params, 'error': f"H1 matched filter error: {e}"})
        print(f"Template {idx}: H1 matched filter error: {e}")
        continue

    # Matched filter for L1
    try:
        snr_l1 = matched_filter(hp, l1_pycbc, psd=l1_psd, low_frequency_cutoff=params['f_lower'])
    except Exception as e:
        mf_failed_templates.append({'params': params, 'error': f"L1 matched filter error: {e}"})
        print(f"Template {idx}: L1 matched filter error: {e}")
        continue

    snr_results.append({
        'params': params,
        'snr_h1': snr_h1,
        'snr_l1': snr_l1
    })

    if (idx+1) % 100 == 0:
        print(f"Processed {idx+1} templates...")

print(f"Matched filtering complete. Successful: {len(snr_results)}, Failed: {len(mf_failed_templates)}.")

# Merge all failed templates for reporting
all_failed_templates = failed_templates + mf_failed_templates

# --- Task 3: Network SNR and Parameter Estimation ---
def compute_chirp_mass(m1, m2):
    return ((m1 * m2) ** (3/5)) / ((m1 + m2) ** (1/5))

def compute_mass_ratio(m1, m2):
    return m2 / m1

print("="*60)
print("Step 4: Combining SNRs to compute network SNR and extract best-fit parameters...")

max_net_snr = -np.inf
best_result = None
all_peaks = []

for idx, result in enumerate(snr_results):
    try:
        snr_h1 = result['snr_h1']
        snr_l1 = result['snr_l1']
        t0 = max(snr_h1.start_time, snr_l1.start_time)
        t1 = min(snr_h1.end_time, snr_l1.end_time)
        if t1 <= t0:
            print(f"Template {idx}: No overlapping time window between H1 and L1 SNRs.")
            continue
        h1_slice = snr_h1.time_slice(t0, t1)
        l1_slice = snr_l1.time_slice(t0, t1)
        net_snr = np.sqrt(np.abs(h1_slice.numpy()**2 + l1_slice.numpy()**2))
        peak_idx = np.argmax(net_snr)
        peak_snr = net_snr[peak_idx]
        peak_time = h1_slice.sample_times[peak_idx]
        all_peaks.append({
            'params': result['params'],
            'peak_net_snr': peak_snr,
            'peak_time': peak_time,
            'idx': idx
        })
        if peak_snr > max_net_snr:
            max_net_snr = peak_snr
            best_result = {
                'params': result['params'],
                'peak_net_snr': peak_snr,
                'peak_time': peak_time,
                'idx': idx
            }
    except Exception as e:
        print(f"Error processing template {idx}: {e}")
        continue

if best_result is None:
    print("No valid network SNR peaks found. Exiting.")
    exit(1)

p = best_result['params']
m1, m2 = p['mass1'], p['mass2']
chirp_mass = compute_chirp_mass(m1, m2)
total_mass = m1 + m2
mass_ratio = compute_mass_ratio(m1, m2)
time_offset = best_result['peak_time']

print("\nBest-fit parameters (maximum network SNR):")
print(f"  Network SNR: {best_result['peak_net_snr']:.2f}")
print(f"  Time offset: {time_offset:.6f} s")
print(f"  Mass1: {m1:.2f} Msun")
print(f"  Mass2: {m2:.2f} Msun")
print(f"  Chirp mass: {chirp_mass:.4f} Msun")
print(f"  Total mass: {total_mass:.2f} Msun")
print(f"  Mass ratio (q = m2/m1): {mass_ratio:.4f}")
print(f"  Spin1z: {p['spin1z']:.2f}")
print(f"  Spin2z: {p['spin2z']:.2f}")

snr_threshold = best_result['peak_net_snr'] - 1.0
close_peaks = [peak for peak in all_peaks if peak['peak_net_snr'] >= snr_threshold]
chirp_masses = []
total_masses = []
mass_ratios = []
time_offsets = []
for peak in close_peaks:
    p = peak['params']
    m1, m2 = p['mass1'], p['mass2']
    chirp_masses.append(compute_chirp_mass(m1, m2))
    total_masses.append(m1 + m2)
    mass_ratios.append(compute_mass_ratio(m1, m2))
    time_offsets.append(peak['peak_time'])
def mean_std(arr):
    arr = np.array(arr)
    return np.mean(arr), np.std(arr)
print("\nParameter uncertainties (templates within 1 SNR of peak):")
cm_mean, cm_std = mean_std(chirp_masses)
tm_mean, tm_std = mean_std(total_masses)
qr_mean, qr_std = mean_std(mass_ratios)
to_mean, to_std = mean_std(time_offsets)
print(f"  Chirp mass: {cm_mean:.4f} ± {cm_std:.4f} Msun")
print(f"  Total mass: {tm_mean:.2f} ± {tm_std:.2f} Msun")
print(f"  Mass ratio: {qr_mean:.4f} ± {qr_std:.4f}")
print(f"  Time offset: {to_mean:.6f} ± {to_std:.6f} s")

best_fit_params = {
    'network_snr': best_result['peak_net_snr'],
    'time_offset': best_result['peak_time'],
    'mass1': best_result['params']['mass1'],
    'mass2': best_result['params']['mass2'],
    'chirp_mass': chirp_mass,
    'total_mass': total_mass,
    'mass_ratio': mass_ratio,
    'spin1z': best_result['params']['spin1z'],
    'spin2z': best_result['params']['spin2z'],
    'uncertainties': {
        'chirp_mass': (cm_mean, cm_std),
        'total_mass': (tm_mean, tm_std),
        'mass_ratio': (qr_mean, qr_std),
        'time_offset': (to_mean, to_std)
    }
}

# --- Task 4: Reporting and Comparison ---
pipeline_end_time = time.time()
total_runtime = pipeline_end_time - pipeline_start_time

published_m1 = 12.0
published_m2 = 7.0

print("\n" + "="*60)
print("===== GW170608 Matched Filtering Analysis Report =====\n")

print("Best-fit parameters (from template bank search):")
print(f"  Mass1:       {best_fit_params['mass1']:.2f} Msun")
print(f"  Mass2:       {best_fit_params['mass2']:.2f} Msun")
print(f"  Chirp mass:  {best_fit_params['chirp_mass']:.4f} Msun")
print(f"  Total mass:  {best_fit_params['total_mass']:.2f} Msun")
print(f"  Mass ratio:  {best_fit_params['mass_ratio']:.4f}")
print(f"  Spin1z:      {best_fit_params['spin1z']:.2f}")
print(f"  Spin2z:      {best_fit_params['spin2z']:.2f}")
print(f"  Time offset: {best_fit_params['time_offset']:.6f} s")
print("\nParameter uncertainties (1 SNR of peak):")
unc = best_fit_params['uncertainties']
print(f"  Chirp mass:  {unc['chirp_mass'][0]:.4f} ± {unc['chirp_mass'][1]:.4f} Msun")
print(f"  Total mass:  {unc['total_mass'][0]:.2f} ± {unc['total_mass'][1]:.2f} Msun")
print(f"  Mass ratio:  {unc['mass_ratio'][0]:.4f} ± {unc['mass_ratio'][1]:.4f}")
print(f"  Time offset: {unc['time_offset'][0]:.6f} ± {unc['time_offset'][1]:.6f} s")

print("\nPublished GW170608 values:")
print(f"  Mass1:       {published_m1:.2f} Msun")
print(f"  Mass2:       {published_m2:.2f} Msun")

print("\nPeak network SNR:")
print(f"  Network SNR: {best_fit_params['network_snr']:.2f}")

if best_fit_params['network_snr'] >= 8:
    snr_significance = "Highly significant (SNR > 8, typical GW detection threshold)"
elif best_fit_params['network_snr'] >= 5:
    snr_significance = "Marginal significance (SNR > 5)"
else:
    snr_significance = "Low significance (SNR < 5)"
print(f"  Significance: {snr_significance}")

print("\nTemplate bank statistics:")
bank_size = len(template_bank)
m1s = np.array([tpl['params']['mass1'] for tpl in template_bank])
m2s = np.array([tpl['params']['mass2'] for tpl in template_bank])
spin1zs = np.array([tpl['params']['spin1z'] for tpl in template_bank])
spin2zs = np.array([tpl['params']['spin2z'] for tpl in template_bank])
print(f"  Total templates: {bank_size}")
print(f"  Mass1 range:     {m1s.min():.2f} – {m1s.max():.2f} Msun")
print(f"  Mass2 range:     {m2s.min():.2f} – {m2s.max():.2f} Msun")
print(f"  Spin1z range:    {spin1zs.min():.2f} – {spin1zs.max():.2f}")
print(f"  Spin2z range:    {spin2zs.min():.2f} – {spin2zs.max():.2f}")

print("\nComputation time:")
print(f"  Total runtime: {total_runtime/60:.2f} minutes ({total_runtime:.1f} seconds)")

print("\nError logs:")
total_failures = 0
if all_failed_templates:
    print(f"  Number of failed templates: {len(all_failed_templates)}")
    total_failures += len(all_failed_templates)
    for i, err in enumerate(all_failed_templates[:5]):
        print(f"    [{i+1}] Params: {err['params']}, Error: {err['error']}")
    if len(all_failed_templates) > 5:
        print(f"    ... and {len(all_failed_templates)-5} more failures.")
else:
    print("  No template generation or filtering failures detected.")

if total_failures == 0:
    print("\nAll steps completed successfully.")
else:
    print(f"\nCompleted with {total_failures} failures. Please review error logs above.")

print("\n===== End of Report =====\n")