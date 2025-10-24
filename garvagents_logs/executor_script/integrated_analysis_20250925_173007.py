# --- Imports ---
import numpy as np
import logging
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation, welch
import os

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("GW170608_analysis")

# --- Section 1: Download LIGO Strain Data ---
print("\n=== 1. Downloading LIGO H1 and L1 strain data for GW170608 ===")
gps_center = 1180922494.5
duration = 64  # seconds
start_time = gps_center - duration / 2
end_time = gps_center + duration / 2
detectors = ['H1', 'L1']
strain_data = {}

for det in detectors:
    print(f"Attempting to fetch data for {det} from {start_time} to {end_time} (GPS)...")
    try:
        ts = TimeSeries.fetch_open_data(det, start_time, end_time)
        strain_data[det] = ts
        print(f"Successfully fetched data for {det}.")
    except Exception as e:
        print(f"Failed to fetch data for {det}: {e}")
        strain_data[det] = None

# Save raw data for reproducibility
os.makedirs("results", exist_ok=True)
for det, ts in strain_data.items():
    if ts is not None:
        ts.write(f"results/{det}_raw_strain.gwf", format='gwf')
print("Raw strain data saved (GWpy GWF format).")

# --- Section 2: Preprocessing (Bandpass + Whitening) ---
print("\n=== 2. Preprocessing: Bandpass filtering and whitening ===")
low_freq = 30
high_freq = 300
preprocessed_data = {}

for det, ts in strain_data.items():
    print(f"\nProcessing {det} data...")
    if ts is None:
        print(f"No data available for {det}, skipping preprocessing.")
        preprocessed_data[det] = None
        continue
    try:
        print(f"Applying {low_freq}-{high_freq} Hz bandpass filter to {det}...")
        ts_bp = ts.bandpass(low_freq, high_freq, filtfilt=True)
        print(f"Whitening {det} data...")
        ts_white = ts_bp.whiten()
        preprocessed_data[det] = ts_white
        print(f"Preprocessing complete for {det}.")
        # Save preprocessed data
        ts_white.write(f"results/{det}_preprocessed_strain.gwf", format='gwf')
    except Exception as e:
        print(f"Error during preprocessing for {det}: {e}")
        preprocessed_data[det] = None

# --- Section 3: Template Bank Generation ---
print("\n=== 3. Generating template bank (IMRPhenomPv2) ===")
m1_vals = np.arange(10, 15, 1)  # 10, 11, 12, 13, 14
m2_vals = np.arange(7, 12, 1)   # 7, 8, 9, 10, 11
spin_vals = [-0.3, 0.0, 0.3]
approximant = "IMRPhenomPv2"
f_lower = 30.0
delta_t = 1.0 / 4096
duration = 16

template_bank = []
total_templates = 0
failed_templates = 0

for m1 in m1_vals:
    for m2 in m2_vals:
        if m1 < m2:
            continue  # Only generate templates with m1 >= m2
        for spin1z in spin_vals:
            for spin2z in spin_vals:
                params = {
                    'mass1': m1,
                    'mass2': m2,
                    'spin1z': spin1z,
                    'spin2z': spin2z,
                    'approximant': approximant,
                    'f_lower': f_lower,
                    'delta_t': delta_t,
                    'duration': duration
                }
                try:
                    hp, hc = get_td_waveform(**params)
                    template_bank.append({
                        'params': params,
                        'hp': hp,
                        'hc': hc
                    })
                    total_templates += 1
                    print(f"Generated template: m1={m1}, m2={m2}, spin1z={spin1z}, spin2z={spin2z}")
                except Exception as e:
                    failed_templates += 1
                    logger.warning(f"Failed to generate template for m1={m1}, m2={m2}, "
                                   f"spin1z={spin1z}, spin2z={spin2z}: {e}")

print(f"Template bank generation complete. {total_templates} templates generated, {failed_templates} failures.")
# Save template bank parameters for reproducibility
np.savez("results/template_bank_params.npz", templates=[t['params'] for t in template_bank])

# --- Section 4: Matched Filtering and Analysis ---
print("\n=== 4. Matched filtering and analysis ===")
GW170608_params = {
    'chirp_mass': 7.9,  # solar masses (approximate, see LIGO/Virgo GWTC-1)
    'mass_ratio': 0.67  # m2/m1 (approximate)
}

# Helper: Convert GWpy TimeSeries to PyCBC TimeSeries
def gwpy_to_pycbc(ts):
    return PyCBC_TimeSeries(ts.value, delta_t=ts.dt, epoch=ts.t0.value)

# Prepare data for PyCBC
data_pycbc = {}
for det in ['H1', 'L1']:
    ts = preprocessed_data.get(det)
    if ts is not None:
        try:
            data_pycbc[det] = gwpy_to_pycbc(ts)
            print(f"Converted {det} data to PyCBC TimeSeries.")
        except Exception as e:
            print(f"Error converting {det} data: {e}")
            data_pycbc[det] = None
    else:
        print(f"No preprocessed data for {det}.")
        data_pycbc[det] = None

# Estimate PSD for each detector using Welch's method
psds = {}
for det, data in data_pycbc.items():
    if data is not None:
        try:
            print(f"Estimating PSD for {det}...")
            psd = welch(data, seg_len=4 * int(1.0 / data.delta_t))
            psd = interpolate(psd, len(data))
            psd = inverse_spectrum_truncation(psd, int(4 / data.delta_t))
            psds[det] = psd
            print(f"PSD estimated for {det}.")
        except Exception as e:
            print(f"Error estimating PSD for {det}: {e}")
            psds[det] = None
    else:
        psds[det] = None

# Matched filtering
results = []
failed_filters = 0

print("Starting matched filtering...")

for idx, template in enumerate(template_bank):
    params = template['params']
    hp = template['hp']
    snr_peaks = {}
    snr_series = {}
    for det in ['H1', 'L1']:
        data = data_pycbc.get(det)
        psd = psds.get(det)
        if data is None or psd is None:
            snr_peaks[det] = None
            snr_series[det] = None
            continue
        try:
            # Ensure template and data are the same length
            if len(hp) > len(data):
                hp_trim = hp[:len(data)]
            elif len(hp) < len(data):
                pad = np.zeros(len(data) - len(hp))
                hp_trim = PyCBC_TimeSeries(np.concatenate([hp.numpy(), pad]), delta_t=hp.delta_t, epoch=hp.start_time)
            else:
                hp_trim = hp

            snr = matched_filter(hp_trim, data, psd=psd, low_frequency_cutoff=30.0)
            snr_series[det] = snr
            snr_peaks[det] = abs(snr).numpy().max()
        except Exception as e:
            print(f"Matched filtering failed for template {idx} ({params}) on {det}: {e}")
            snr_peaks[det] = None
            snr_series[det] = None
            failed_filters += 1

    # Compute network SNR (quadrature sum of peak SNRs)
    if snr_peaks['H1'] is not None and snr_peaks['L1'] is not None:
        network_snr = np.sqrt(snr_peaks['H1']**2 + snr_peaks['L1']**2)
    else:
        network_snr = None

    # Compute chirp mass and mass ratio
    m1 = params['mass1']
    m2 = params['mass2']
    chirp_mass = ((m1 * m2) ** (3.0/5.0)) / ((m1 + m2) ** (1.0/5.0))
    mass_ratio = m2 / m1

    results.append({
        'params': params,
        'chirp_mass': chirp_mass,
        'mass_ratio': mass_ratio,
        'snr_peaks': snr_peaks,
        'network_snr': network_snr,
        'snr_series': snr_series
    })

print(f"Matched filtering complete. {failed_filters} filter failures out of {len(template_bank)*2} filterings.")

# Find the template with the highest network SNR
valid_results = [r for r in results if r['network_snr'] is not None]
if valid_results:
    best_result = max(valid_results, key=lambda r: r['network_snr'])
    best_params = best_result['params']
    best_chirp_mass = best_result['chirp_mass']
    best_mass_ratio = best_result['mass_ratio']
    best_network_snr = best_result['network_snr']
    print("\n=== Best-fit template parameters ===")
    print(f"  mass1: {best_params['mass1']} M_sun")
    print(f"  mass2: {best_params['mass2']} M_sun")
    print(f"  spin1z: {best_params['spin1z']}")
    print(f"  spin2z: {best_params['spin2z']}")
    print(f"  Chirp mass: {best_chirp_mass:.2f} M_sun")
    print(f"  Mass ratio: {best_mass_ratio:.2f}")
    print(f"  Peak network SNR: {best_network_snr:.2f}")
    print("\nComparison with published GW170608 values:")
    print(f"  Published chirp mass: {GW170608_params['chirp_mass']} M_sun")
    print(f"  Published mass ratio: {GW170608_params['mass_ratio']}")
else:
    print("No valid matched filter results found.")

# Report template bank statistics
print("\n=== Template bank statistics ===")
print(f"  Total templates: {len(template_bank)}")
print(f"  Total matched filter attempts: {len(template_bank)*2}")
print(f"  Matched filter failures: {failed_filters}")
print(f"  Successful matched filterings: {len(template_bank)*2 - failed_filters}")
if valid_results:
    print(f"  Peak network SNR: {best_network_snr:.2f}")
else:
    print("  No valid network SNR found.")

# Save results for further analysis
import pickle
with open("results/matched_filter_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("Matched filter results saved to results/matched_filter_results.pkl")