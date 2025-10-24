# --- Imports ---
import os
import traceback
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# --- Configuration ---
GW190521_GPS = 1242442967.4
DURATION = 64  # seconds (±32s)
START = GW190521_GPS - DURATION / 2
END = GW190521_GPS + DURATION / 2
DETECTORS = ['H1', 'L1']

# Output directories
PLOT_DIR = "gw190521_plots"
QTRANSFORM_DIR = "gw190521_qtransforms"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(QTRANSFORM_DIR, exist_ok=True)

# --- 1. Download GW190521 Strain Data ---
print("="*60)
print("STEP 1: Downloading GW190521 strain data from GWOSC")
print("="*60)
gw190521_strain_data = {}

for det in DETECTORS:
    try:
        print(f"Fetching {DURATION}s of strain data for {det} from GPS {START} to {END}...")
        ts = TimeSeries.fetch_open_data(det, START, END, cache=True)
        gw190521_strain_data[det] = ts
        print(f"  Success: {det} data downloaded.")
    except Exception as e:
        print(f"  ERROR: Failed to download data for {det}: {e}")
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
        print(f"Skipping {det}: No data available for filtering.")
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

print("\nBandpass filtering complete. Filtered data is stored in 'filtered_gw190521_strain'.")

# --- 3. Plot Filtered Strain (Time Domain) ---
print("\n" + "="*60)
print("STEP 3: Plotting filtered GW190521 strain data (±0.2s around merger)")
print("="*60)
window = 0.2  # ±0.2 seconds

for det, ts in filtered_gw190521_strain.items():
    if ts is None:
        print(f"Skipping {det}: No filtered data available for time-domain plot.")
        continue
    try:
        print(f"Plotting time-domain strain for {det} (±{window}s around merger)...")
        ts_window = ts.crop(GW190521_GPS - window, GW190521_GPS + window)
        times = ts_window.times.value - GW190521_GPS
        plt.figure(figsize=(10, 4))
        plt.plot(times, ts_window.value, label=f"{det} strain")
        plt.title(f"{det} Filtered Strain | GW190521 | ±{window}s around Merger")
        plt.xlabel("Time (s) since merger")
        plt.ylabel("Strain")
        plt.axvline(0, color='r', linestyle='--', label='Merger Time')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"GW190521_{det}_strain.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  Success: Time-domain plot for {det} saved to {plot_path}")
    except Exception as e:
        print(f"  ERROR: Failed to plot time-domain data for {det}: {e}")
        traceback.print_exc()

# --- 4. Q-transform (Q Spectrogram) ---
print("\n" + "="*60)
print("STEP 4: Generating Q-transform (Q spectrogram) for GW190521 (±2s around merger)")
print("="*60)
q_window = 2  # seconds before and after merger

for det, ts in filtered_gw190521_strain.items():
    if ts is None:
        print(f"Skipping {det}: No filtered data available for Q-transform plot.")
        continue
    try:
        print(f"Generating Q-transform for {det} (±{q_window}s around merger)...")
        ts_window = ts.crop(GW190521_GPS - q_window, GW190521_GPS + q_window)
        q = ts_window.q_transform(outseg=(GW190521_GPS - q_window, GW190521_GPS + q_window))
        fig = q.plot(figsize=(10, 6), vmin=1e-24, vmax=1e-21)
        ax = fig.gca()
        ax.axvline(GW190521_GPS, color='r', linestyle='--', label='Merger Time')
        ax.set_title(f"{det} Q-transform | GW190521 | ±{q_window}s around Merger")
        ax.set_xlabel("Time (GPS)")
        ax.set_ylabel("Frequency (Hz)")
        ax.legend()
        plt.tight_layout()
        q_path = os.path.join(QTRANSFORM_DIR, f"GW190521_{det}_qtransform.png")
        plt.savefig(q_path)
        plt.close()
        print(f"  Success: Q-transform plot for {det} saved to {q_path}")
    except Exception as e:
        print(f"  ERROR: Failed to generate Q-transform for {det}: {e}")
        traceback.print_exc()

print("\nAll tasks complete. Results saved in 'gw190521_plots/' and 'gw190521_qtransforms/' directories.")