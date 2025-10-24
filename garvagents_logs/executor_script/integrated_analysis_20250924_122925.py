# --- Imports ---
import os
import traceback
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# --- Configuration ---
GW190521_GPS = 1242442967.4
START_TIME = GW190521_GPS - 32
END_TIME = GW190521_GPS + 32
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
        print(f"Fetching {det} strain data for GW190521 from {START_TIME} to {END_TIME}...")
        ts = TimeSeries.fetch_open_data(det, START_TIME, END_TIME, cache=True)
        gw190521_strain_data[det] = ts
        print(f"  Success: {det} data downloaded. Duration: {ts.duration.value} seconds.")
    except Exception as e:
        print(f"  ERROR: Failed to fetch {det} data: {e}")
        traceback.print_exc()
        gw190521_strain_data[det] = None

print("\nData loading complete. Strain data is stored in the 'gw190521_strain_data' dictionary.")

# --- 2. Apply Bandpass Filter ---
print("\n" + "="*60)
print("STEP 2: Filtering GW190521 strain data (35–350 Hz bandpass)")
print("="*60)
filtered_gw190521_strain = {}

for det, ts in gw190521_strain_data.items():
    if ts is None:
        print(f"Skipping {det}: No strain data available.")
        filtered_gw190521_strain[det] = None
        continue
    try:
        print(f"Applying bandpass filter (35–350 Hz) to {det} data...")
        filtered_ts = ts.bandpass(35, 350)
        filtered_gw190521_strain[det] = filtered_ts
        print(f"  Success: {det} data filtered.")
    except Exception as e:
        print(f"  ERROR: Failed to filter {det} data: {e}")
        traceback.print_exc()
        filtered_gw190521_strain[det] = None

print("\nFiltering complete. Filtered data is stored in 'filtered_gw190521_strain'.")

# --- 3. Plot Filtered Strain (Time Domain) ---
print("\n" + "="*60)
print("STEP 3: Plotting filtered GW190521 strain data (±0.2s around merger)")
print("="*60)
window = 0.2  # ±0.2 seconds

for det, ts in filtered_gw190521_strain.items():
    if ts is None:
        print(f"Skipping {det}: No filtered data available.")
        continue
    try:
        print(f"Plotting filtered strain for {det} around the merger...")
        ts_window = ts.crop(GW190521_GPS - window, GW190521_GPS + window)
        fig = ts_window.plot()
        ax = fig.gca()
        ax.set_title(f"GW190521 {det} | Filtered Strain ±{window}s around Merger")
        ax.set_xlabel("Time [s] (GPS)")
        ax.set_ylabel("Strain")
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"GW190521_{det}_strain.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  Success: Time-domain plot for {det} saved to {plot_path}")
    except Exception as e:
        print(f"  ERROR: Failed to plot strain for {det}: {e}")
        traceback.print_exc()

# --- 4. Plot Spectrogram (Time–Frequency) using scipy.signal.spectrogram ---
print("\n" + "="*60)
print("STEP 4: Generating time–frequency spectrograms for GW190521 (±2s around merger)")
print("="*60)
spec_window = 2.0  # ±2 seconds

for det, ts in filtered_gw190521_strain.items():
    if ts is None:
        print(f"Skipping {det}: No filtered data available.")
        continue
    try:
        print(f"Generating spectrogram for {det} around the merger using scipy.signal.spectrogram...")
        ts_window = ts.crop(GW190521_GPS - spec_window, GW190521_GPS + spec_window)
        data = ts_window.value
        fs = ts_window.sample_rate.value

        # Parameters for spectrogram
        nperseg = int(0.125 * fs)  # 0.125s window
        noverlap = int(0.0625 * fs)  # 0.0625s overlap

        # Compute spectrogram
        f, t, Sxx = spectrogram(
            data,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            mode='psd'
        )

        # Convert time axis to GPS time
        t_gps = t + ts_window.t0.value

        # Plot
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(t_gps, f, Sxx, shading='auto', norm='log', vmin=1e-24, vmax=1e-21)
        plt.colorbar(label='PSD [strain^2/Hz]')
        plt.title(f"GW190521 {det} | Spectrogram ±{spec_window}s around Merger")
        plt.xlabel("Time [s] (GPS)")
        plt.ylabel("Frequency [Hz]")
        plt.ylim(30, 400)
        plt.tight_layout()
        spec_path = os.path.join(SPECTROGRAM_DIR, f"GW190521_{det}_spectrogram.png")
        plt.savefig(spec_path)
        plt.close()
        print(f"  Success: Spectrogram for {det} saved to {spec_path}")
    except Exception as e:
        print(f"  ERROR: Failed to generate spectrogram for {det}: {e}")
        traceback.print_exc()

print("\nAll tasks complete. Results saved in 'gw190521_plots/' and 'gw190521_spectrograms/' directories.")