# GW170608 Strain Data Download, Filtering, and Analysis
# ------------------------------------------------------
# This script downloads LIGO H1 and L1 strain data for GW170608,
# applies a bandpass filter, and visualizes the event in both
# the time and time-frequency domains.

# =========================
# Imports and Configuration
# =========================
import sys
import traceback
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
# GW170608 event GPS time
MERGER_GPS = 1180922494.5
# Data interval (±128s)
START_TIME = MERGER_GPS - 128
END_TIME = MERGER_GPS + 128
# Detectors to analyze
DETECTORS = ['H1', 'L1']
# Bandpass filter range
BANDPASS_LOW = 35
BANDPASS_HIGH = 350
# Plotting window (±2s around merger)
PLOT_WINDOW = 2  # seconds

# =========================
# 1. Data Download
# =========================
print("="*60)
print("STEP 1: Downloading strain data from GWOSC...")
strain_data = {}

for det in DETECTORS:
    try:
        print(f"  Fetching data for {det} ({START_TIME} to {END_TIME})...")
        ts = TimeSeries.fetch_open_data(det, START_TIME, END_TIME, cache=True)
        strain_data[det] = ts
        print(f"    Success: {det} data downloaded.")
    except Exception as e:
        print(f"    ERROR: Failed to fetch data for {det}: {e}")
        traceback.print_exc()
        strain_data[det] = None

print("STEP 1 complete.\n")

# =========================
# 2. Bandpass Filtering
# =========================
print("="*60)
print(f"STEP 2: Applying bandpass filter ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)...")
filtered_strain_data = {}

for det in DETECTORS:
    ts = strain_data.get(det)
    if ts is None:
        print(f"  WARNING: No data for {det}; skipping filtering.")
        filtered_strain_data[det] = None
        continue
    try:
        print(f"  Filtering {det}...")
        filtered_ts = ts.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
        filtered_strain_data[det] = filtered_ts
        print(f"    Success: {det} filtered.")
    except Exception as e:
        print(f"    ERROR: Bandpass filtering failed for {det}: {e}")
        traceback.print_exc()
        filtered_strain_data[det] = None

print("STEP 2 complete.\n")

# =========================
# 3. Time-Domain Plotting
# =========================
print("="*60)
print(f"STEP 3: Plotting time-domain strain data (±{PLOT_WINDOW}s around merger)...")

plot_start = MERGER_GPS - PLOT_WINDOW
plot_end = MERGER_GPS + PLOT_WINDOW

for det in DETECTORS:
    ts = filtered_strain_data.get(det)
    if ts is None:
        print(f"  WARNING: No filtered data for {det}; skipping plot.")
        continue
    try:
        print(f"  Plotting {det}...")
        ts_zoom = ts.crop(plot_start, plot_end)
        plt.figure(figsize=(10, 4))
        plt.plot(ts_zoom.times.value, ts_zoom.value, label=f'{det} strain')
        plt.title(f'{det} Strain Data (Bandpassed {BANDPASS_LOW}–{BANDPASS_HIGH} Hz)\n'
                  f'GW170608 (GPS {MERGER_GPS} ± {PLOT_WINDOW} s)')
        plt.xlabel('GPS Time [s]')
        plt.ylabel('Strain')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"    Plot for {det} complete.")
    except Exception as e:
        print(f"    ERROR: Failed to plot {det}: {e}")
        traceback.print_exc()

print("STEP 3 complete.\n")

# =========================
# 4. Q-transform Spectrograms
# =========================
print("="*60)
print(f"STEP 4: Generating Q-transform spectrograms (±{PLOT_WINDOW}s around merger)...")

spec_start = MERGER_GPS - PLOT_WINDOW
spec_end = MERGER_GPS + PLOT_WINDOW

for det in DETECTORS:
    ts = filtered_strain_data.get(det)
    if ts is None:
        print(f"  WARNING: No filtered data for {det}; skipping Q-transform.")
        continue
    try:
        print(f"  Computing Q-transform for {det}...")
        ts_zoom = ts.crop(spec_start, spec_end)
        q = ts_zoom.q_transform()
        fig = q.plot(figsize=(10, 6))
        ax = fig.gca()
        ax.set_title(f'{det} Q-transform Spectrogram\n'
                     f'GW170608 (GPS {MERGER_GPS} ± {PLOT_WINDOW} s)')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('GPS Time [s]')
        plt.tight_layout()
        plt.show()
        print(f"    Q-transform spectrogram for {det} complete.")
    except Exception as e:
        print(f"    ERROR: Failed Q-transform for {det}: {e}")
        traceback.print_exc()

print("STEP 4 complete.\n")
print("="*60)
print("All analysis steps complete.")