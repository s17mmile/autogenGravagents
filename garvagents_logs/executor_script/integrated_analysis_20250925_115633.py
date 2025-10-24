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

event_gps = 1180922494.5
delta_t = 32  # seconds before and after
start = event_gps - delta_t
end = event_gps + delta_t

detectors = ['H1', 'L1']
strain_data = {}
preprocessed_data = {}

for det in detectors:
    print(f"\nFetching {det} strain data from {start} to {end} (GPS)...")
    try:
        strain = TimeSeries.fetch_open_data(det, start, end, cache=True)
        print(f"  {det} data fetched: {len(strain)} samples, dt={strain.dt.value if hasattr(strain.dt, 'value') else strain.dt:.5f} s")
        strain_data[det] = strain
    except Exception as e:
        print(f"  ERROR: Failed to fetch {det} data: {e}")
        strain_data[det] = None
        continue

    # Preprocessing: bandpass and whitening
    try:
        print(f"  Applying bandpass filter (30-300 Hz) to {det}...")
        bp = strain.bandpass(30, 300)
        print(f"  Whitening {det} data...")
        bp_white = bp.whiten()
        preprocessed_data[det] = bp_white
        print(f"  {det} preprocessing complete.")
    except Exception as e:
        print(f"  ERROR: Preprocessing failed for {det}: {e}")
        preprocessed_data[det] = None

h1_strain_processed = preprocessed_data.get('H1', None)
l1_strain_processed = preprocessed_data.get('L1', None)

if h1_strain_processed is None or l1_strain_processed is None:
    print("ERROR: Preprocessed H1/L1 data not found. Exiting.")
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

template_bank = []
failed_templates = []

total = 0
success = 0
for m1 in m1_vals:
    for m2 in m2_vals:
        if m2 > m1:
            continue  # Only physical systems
        for chi1 in spin_vals:
            for chi2 in spin_vals:
                params = {
                    'mass1': float(m1),
                    'mass2': float(m2),
                    'spin1z': float(chi1),
                    'spin2z': float(chi2),
                    'approximant': 'IMRPhenomPv2',
                    'f_lower': f_lower,
                    'delta_t': 1.0 / sample_rate
                }
                total += 1
                try:
                    hp, hc = get_td_waveform(
                        approximant=params['approximant'],
                        mass1=params['mass1'],
                        mass2=params['mass2'],
                        spin1z=params['spin1z'],
                        spin2z=params['spin2z'],
                        delta_t=params['delta_t'],
                        f_lower=params['f_lower'],
                        duration=duration
                    )
                    # Store only parameters, not waveform data, for identification
                    template_bank.append({'params': params})
                    success += 1
                    if success % 100 == 0:
                        print(f"  {success} templates generated...")
                except Exception as e:
                    failed_templates.append({'params': params, 'error': str(e)})
                    print(f"  ERROR: Failed for params {params}: {e}")

print(f"\nTemplate bank generation complete.")
print(f"  Total attempted: {total}")
print(f"  Successfully generated: {success}")
print(f"  Failed: {len(failed_templates)}")

if not template_bank:
    print("ERROR: No templates generated. Exiting.")
    exit(1)

# --- Step 3: Matched Filtering and Network SNR ---
print("="*60)
print("Step 3: Matched filtering and network SNR computation")

def gwpy_to_pycbc(ts):
    return PyCBC_TimeSeries(ts.value, delta_t=ts.dt.value if hasattr(ts.dt, 'value') else ts.dt, epoch=ts.t0.value if hasattr(ts.t0, 'value') else ts.t0)

h1_pycbc = gwpy_to_pycbc(h1_strain_processed)
l1_pycbc = gwpy_to_pycbc(l1_strain_processed)

# Estimate PSDs for both detectors
print("Estimating PSDs for H1 and L1...")
psd_len = 4  # seconds
psd_N = int(psd_len / h1_pycbc.delta_t)
h1_psd = welch(h1_pycbc, seg_len=psd_N)
l1_psd = welch(l1_pycbc, seg_len=psd_N)
print("  PSD estimation complete.")

snr_results = []
snr_failures = []

print("Starting matched filtering for all templates...")
for idx, tpl in enumerate(template_bank):
    params = tpl['params']
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
            duration=duration
        )
        # Resize template to match data length
        if len(hp) > len(h1_pycbc):
            hp = hp.crop(0, len(h1_pycbc) * h1_pycbc.delta_t - hp.duration)
            hp.resize(len(h1_pycbc))
        elif len(hp) < len(h1_pycbc):
            hp.resize(len(h1_pycbc))

        # Matched filter for H1
        snr_h1 = matched_filter(hp, h1_pycbc, psd=h1_psd, low_frequency_cutoff=params['f_lower'])
        peak_snr_h1 = abs(snr_h1).numpy().max()

        # Matched filter for L1
        snr_l1 = matched_filter(hp, l1_pycbc, psd=l1_psd, low_frequency_cutoff=params['f_lower'])
        peak_snr_l1 = abs(snr_l1).numpy().max()

        # Network SNR
        network_snr = np.sqrt(peak_snr_h1**2 + peak_snr_l1**2)

        snr_results.append({
            'params': params,
            'peak_snr_h1': peak_snr_h1,
            'peak_snr_l1': peak_snr_l1,
            'network_snr': network_snr
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(template_bank)} templates...")

    except Exception as e:
        snr_failures.append({'params': params, 'error': str(e)})
        print(f"  ERROR: SNR computation failed for params {params}: {e}")

print(f"\nMatched filtering complete. {len(snr_results)} templates processed, {len(snr_failures)} failures.")

# Identify template with peak network SNR
if snr_results:
    best_idx = np.argmax([r['network_snr'] for r in snr_results])
    best_fit_params = snr_results[best_idx]
    print(f"\nBest-fit template found with network SNR = {best_fit_params['network_snr']:.2f}")
else:
    best_fit_params = None
    print("No successful SNR results found.")
    exit(1)

# --- Step 4: Parameter Estimation and Reporting ---
print("="*60)
print("Step 4: Parameter estimation and reporting")

published_m1 = 12.0
published_m2 = 7.0

try:
    report_start_time = time.time()
    # Extract best-fit parameters
    params = best_fit_params['params']
    m1 = params['mass1']
    m2 = params['mass2']
    spin1z = params['spin1z']
    spin2z = params['spin2z']
    peak_snr_h1 = best_fit_params['peak_snr_h1']
    peak_snr_l1 = best_fit_params['peak_snr_l1']
    peak_network_snr = best_fit_params['network_snr']

    # Derived parameters
    total_mass = m1 + m2
    mass_ratio = m2 / m1
    chirp_mass = ((m1 * m2) ** (3/5)) / (total_mass ** (1/5))

    # Estimate time offset (peak SNR time)
    hp, _ = get_td_waveform(
        approximant=params['approximant'],
        mass1=m1,
        mass2=m2,
        spin1z=spin1z,
        spin2z=spin2z,
        delta_t=params['delta_t'],
        f_lower=params['f_lower'],
        duration=duration
    )
    if len(hp) > len(h1_pycbc):
        hp = hp.crop(0, len(h1_pycbc) * h1_pycbc.delta_t - hp.duration)
        hp.resize(len(h1_pycbc))
    elif len(hp) < len(h1_pycbc):
        hp.resize(len(h1_pycbc))
    snr_h1_series = matched_filter(hp, h1_pycbc, psd=h1_psd, low_frequency_cutoff=params['f_lower'])
    peak_idx = np.argmax(abs(snr_h1_series))
    time_offset = snr_h1_series.sample_times[peak_idx] - h1_pycbc.start_time

    # Estimate uncertainties: templates within 1% of peak network SNR
    snr_thresh = 0.99 * peak_network_snr
    close_templates = [r for r in snr_results if r['network_snr'] >= snr_thresh]
    m1s = np.array([r['params']['mass1'] for r in close_templates])
    m2s = np.array([r['params']['mass2'] for r in close_templates])
    chis1 = np.array([r['params']['spin1z'] for r in close_templates])
    chis2 = np.array([r['params']['spin2z'] for r in close_templates])
    chirp_masses = ((m1s * m2s) ** (3/5)) / ((m1s + m2s) ** (1/5))
    total_masses = m1s + m2s
    mass_ratios = m2s / m1s

    def err(arr):
        return np.std(arr) if len(arr) > 1 else 0.0

    m1_err = err(m1s)
    m2_err = err(m2s)
    chirp_mass_err = err(chirp_masses)
    total_mass_err = err(total_masses)
    mass_ratio_err = err(mass_ratios)
    spin1z_err = err(chis1)
    spin2z_err = err(chis2)

    total_templates = len(template_bank)
    computation_time = time.time() - report_start_time

    print("\n===== GW170608 Parameter Estimation Report =====")
    print(f"Best-fit template parameters:")
    print(f"  mass1      = {m1:.2f} ± {m1_err:.2f} M☉")
    print(f"  mass2      = {m2:.2f} ± {m2_err:.2f} M☉")
    print(f"  spin1z     = {spin1z:.2f} ± {spin1z_err:.2f}")
    print(f"  spin2z     = {spin2z:.2f} ± {spin2z_err:.2f}")
    print(f"  chirp mass = {chirp_mass:.2f} ± {chirp_mass_err:.2f} M☉")
    print(f"  total mass = {total_mass:.2f} ± {total_mass_err:.2f} M☉")
    print(f"  mass ratio = {mass_ratio:.2f} ± {mass_ratio_err:.2f}")
    print(f"  time offset (from data start) = {time_offset:.4f} s")
    print(f"\nPeak SNRs:")
    print(f"  H1         = {peak_snr_h1:.2f}")
    print(f"  L1         = {peak_snr_l1:.2f}")
    print(f"  Network    = {peak_network_snr:.2f}")
    print(f"\nTemplate bank statistics:")
    print(f"  Total templates: {total_templates}")
    print(f"  SNR computation time: {computation_time:.1f} s")
    print(f"\nComparison to published GW170608 values:")
    print(f"  Published m1 ≈ {published_m1:.1f} M☉, m2 ≈ {published_m2:.1f} M☉")
    print(f"  Δm1 = {m1 - published_m1:+.2f} M☉, Δm2 = {m2 - published_m2:+.2f} M☉")
    print("===============================================")

    if failed_templates or snr_failures:
        print("\nError logs:")
        if failed_templates:
            print(f"  Template generation failures: {len(failed_templates)}")
            for i, err in enumerate(failed_templates[:5]):
                print(f"    [{i+1}] Params: {err['params']}, Error: {err['error']}")
            if len(failed_templates) > 5:
                print(f"    ... and {len(failed_templates)-5} more failures.")
        if snr_failures:
            print(f"  SNR computation failures: {len(snr_failures)}")
            for i, err in enumerate(snr_failures[:5]):
                print(f"    [{i+1}] Params: {err['params']}, Error: {err['error']}")
            if len(snr_failures) > 5:
                print(f"    ... and {len(snr_failures)-5} more failures.")
    else:
        print("\nNo template generation or SNR computation failures detected.")

except Exception as e:
    print(f"ERROR during parameter estimation and reporting: {e}")

print("\nWorkflow complete.")