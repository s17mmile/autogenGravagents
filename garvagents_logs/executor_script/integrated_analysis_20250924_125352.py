# --- Imports ---
import os
import traceback
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# --- Configuration ---
GW170608_GPS = 1180922494.5
WINDOW = 32  # seconds before and after
START = GW170608_GPS - WINDOW
END = GW170608_GPS + WINDOW
DETECTORS = ['H1', 'L1']

# Output directories
PLOT_DIR = "gw170608_plots"
QTRANSFORM_DIR = "gw170608_qtransforms"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(QTRANSFORM_DIR, exist_ok=True)

# --- 1. Download GW170608 Strain Data ---
print("="*60)
print("STEP 1: Downloading GW170608 strain data from GWOSC")
print("="*60)
gw170608_strain_data = {}

for det in DETECTORS:
    try:
        print(f"Fetching strain data for {det} from {START} to {END} (GPS)...")
        ts = TimeSeries.fetch_open_data(det, START, END, cache=True)
        gw170608_strain_data[det] = ts
        print(f"  Success: {det} data downloaded.")
    except Exception as e:
        print(f"  ERROR: Failed to download data for {det}: {e}")
        traceback.print_exc()
        gw170608_strain_data[det] = None

print("\nData download complete. Strain data is stored in 'gw170608_strain_data'.")

# --- 2. Apply Bandpass Filter ---
print("\n" + "="*60)
print("STEP 2: Filtering GW170608 strain data (35–350 Hz bandpass)")
print("="*60)
filtered_gw170608_strain = {}

for det, ts in gw170608_strain_data.items():
    if ts is None:
        print(f"Skipping {det}: No data available for filtering.")
        filtered_gw170608_strain[det] = None
        continue
    try:
        print(f"Applying 35–350 Hz bandpass filter to {det} data...")
        filtered_ts = ts.bandpass(35, 350)
        filtered_gw170608_strain[det] = filtered_ts
        print(f"  Success: {det} data filtered.")
    except Exception as e:
        print(f"  ERROR: Failed to filter {det} data: {e}")
        traceback.print_exc()
        filtered_gw170608_strain[det] = None

print("\nFiltering complete. Filtered data is stored in 'filtered_gw170608_strain'.")

# --- 3. Plot Filtered Strain (Time Domain) ---
print("\n" + "="*60)
print("STEP 3: Plotting filtered GW170608 strain data (±0.2s around merger)")
print("="*60)
plot_window = 0.2  # ±0.2 seconds

for det, ts in filtered_gw170608_strain.items():
    if ts is None:
        print(f"Skipping {det}: No filtered data available for time-domain plot.")
        continue
    try:
        print(f"Plotting time-domain strain for {det} (±{plot_window}s around merger)...")
        ts_window = ts.crop(GW170608_GPS - plot_window, GW170608_GPS + plot_window)
        time_rel = ts_window.times.value - GW170608_GPS
        plt.figure(figsize=(10, 4))
        plt.plot(time_rel, ts_window.value, label=f'{det} strain')
        plt.axvline(0, color='r', linestyle='--', label='Merger Time')
        plt.title(f"{det} Filtered Strain | GW170608 | ±{plot_window}s around Merger")
        plt.xlabel("Time (s) relative to merger")
        plt.ylabel("Strain")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"GW170608_{det}_strain.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  Success: Time-domain plot for {det} saved to {plot_path}")
    except Exception as e:
        print(f"  ERROR: Failed to plot time-domain strain for {det}: {e}")
        traceback.print_exc()

# --- 4. Q-transform (Q Spectrogram) ---
print("\n" + "="*60)
print("STEP 4: Generating Q-transform (Q spectrogram) for GW170608 (±2s around merger)")
print("="*60)
q_window = 2  # seconds before and after merger

for det, ts in filtered_gw170608_strain.items():
    if ts is None:
        print(f"Skipping {det}: No filtered data available for Q-transform.")
        continue
    try:
        print(f"Generating Q-transform for {det} (±{q_window}s around merger)...")
        ts_window = ts.crop(GW170608_GPS - q_window, GW170608_GPS + q_window)
        q = ts_window.q_transform(outseg=(GW170608_GPS - q_window, GW170608_GPS + q_window))
        fig = q.plot(figsize=(10, 6))
        ax = fig.gca()
        ax.axvline(GW170608_GPS, color='r', linestyle='--', label='Merger Time')
        ax.set_title(f"{det} Q-transform | GW170608 | ±{q_window}s around Merger")
        ax.set_xlabel("Time (GPS)")
        ax.set_ylabel("Frequency [Hz]")
        ax.legend()
        plt.tight_layout()
        q_path = os.path.join(QTRANSFORM_DIR, f"GW170608_{det}_qtransform.png")
        plt.savefig(q_path)
        plt.close()
        print(f"  Success: Q-transform plot for {det} saved to {q_path}")
    except Exception as e:
        print(f"  ERROR: Failed to generate Q-transform for {det}: {e}")
        traceback.print_exc()

print("\nAll tasks complete. Results saved in 'gw170608_plots/' and 'gw170608_qtransforms/' directories.")