# --- Imports ---
import os
import traceback
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# --- Configuration ---
GW190521_GPS = 1242442967.4
DURATION = 64  # seconds (±32s)
START = GW190521_GPS - DURATION / 2
END = GW190521_GPS + DURATION / 2
DETECTORS = ['H1', 'L1']

# Output directories
PLOT_DIR = "gw190521_plots"
SPECTROGRAM_DIR = "gw190521_spectrograms"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

# --- 1. Download GW190521 Strain Data ---
print("="*60)
print("STEP 1: Downloading GW190521 strain data from GWOSC")
print("="*60)
gw190521_strain_data = {}

for det in DETECTORS:
    try:
        print(f"Fetching {DURATION} seconds of strain data for {det} from {START} to {END} (GPS)...")
        ts = TimeSeries.fetch_open_data(det, START, END, cache=True)
        gw190521_strain_data[det] = ts
        print(f"  Success: {det} data downloaded.")
    except Exception as e:
        print(f"  ERROR: Failed to fetch data for {det}: {e}")
        traceback.print_exc()
        gw190521_strain_data[det] = None

print("\nData download complete. Strain data is stored in 'gw190521_strain_data'.")

# --- 2. Apply Bandpass Filter ---
print("\n" + "="*60)
print("STEP 2: Filtering GW190521 strain data (35–350 Hz bandpass)")
print("="*60)
filtered_gw190521_strain = {}

for det, ts in gw190521_strain_data.items():
    if ts is None:
        print(f"Skipping {det}: No data available to filter.")
        filtered_gw190521_strain[det] = None
        continue
    try:
        print(f"Applying 35–350 Hz bandpass filter to {det} strain data...")
        filtered_ts = ts.bandpass(35, 350)
        filtered_gw190521_strain[det] = filtered_ts
        print(f"  Success: {det} data filtered.")
    except Exception as e:
        print(f"  ERROR: Failed to filter {det} data: {e}")
        traceback.print_exc()
        filtered_gw190521_strain[det] = None

print("\nFiltering complete. Filtered strain data is stored in 'filtered_gw190521_strain'.")

# --- 3. Plot Filtered Strain (Time Domain) ---
print("\n" + "="*60)
print("STEP 3: Plotting filtered GW190521 strain data (±0.2s around merger)")
print("="*60)
window = 0.2  # ±0.2 seconds

for det, ts in filtered_gw190521_strain.items():
    if ts is None:
        print(f"Skipping {det}: No filtered data available for plotting.")
        continue
    try:
        print(f"Plotting time-domain strain for {det} (±{window}s around merger)...")
        ts_window = ts.crop(GW190521_GPS - window, GW190521_GPS + window)
        times = ts_window.times.value - GW190521_GPS
        plt.figure(figsize=(10, 4))
        plt.plot(times, ts_window.value, label=f"{det} filtered strain")
        plt.title(f"{det} Filtered Strain | GW190521 | ±{window}s around Merger")
        plt.xlabel("Time (s) relative to merger")
        plt.ylabel("Strain")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"GW190521_{det}_strain.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  Success: Time-domain plot for {det} saved to {plot_path}")
    except Exception as e:
        print(f"  ERROR: Failed to plot time-domain strain for {det}: {e}")
        traceback.print_exc()

# --- 4. Plot Spectrogram (Time–Frequency) ---
print("\n" + "="*60)
print("STEP 4: Generating time–frequency spectrograms for GW190521 (±2s around merger)")
print("="*60)
spec_window = 2.0  # ±2 seconds

for det, ts in filtered_gw190521_strain.items():
    if ts is None:
        print(f"Skipping {det}: No filtered data available for spectrogram.")
        continue
    try:
        print(f"Generating spectrogram for {det} (±{spec_window}s around merger)...")
        ts_window = ts.crop(GW190521_GPS - spec_window, GW190521_GPS + spec_window)
        # Use scipy.signal.spectrogram instead of GWpy's spectrogram
        data = ts_window.value
        fs = ts_window.sample_rate.value
        nperseg = int(0.125 * fs)  # 0.125s window
        noverlap = int(nperseg * 0.5)  # 50% overlap
        f, t, Sxx = spectrogram(
            data, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='psd'
        )
        # Only plot frequencies in 30-400 Hz
        freq_mask = (f >= 30) & (f <= 400)
        f_plot = f[freq_mask]
        Sxx_plot = Sxx[freq_mask, :]
        # Time axis: relative to GW190521_GPS
        t_plot = t + (ts_window.t0.value - GW190521_GPS)
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t_plot, f_plot, 10 * np.log10(Sxx_plot), shading='auto', cmap='viridis')
        plt.title(f"{det} Spectrogram | GW190521 | ±{spec_window}s around Merger")
        plt.xlabel("Time (s) relative to merger")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(30, 400)
        plt.colorbar(label='PSD [dB]')
        plt.tight_layout()
        spec_path = os.path.join(SPECTROGRAM_DIR, f"GW190521_{det}_spectrogram.png")
        plt.savefig(spec_path)
        plt.close()
        print(f"  Success: Spectrogram for {det} saved to {spec_path}")
    except Exception as e:
        print(f"  ERROR: Failed to generate spectrogram for {det}: {e}")
        traceback.print_exc()

print("\nAll tasks complete. Results saved in 'gw190521_plots/' and 'gw190521_spectrograms/' directories.")