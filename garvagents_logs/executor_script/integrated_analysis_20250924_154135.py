# GW150914 Strain Data Download, Filtering, and Q-transform Analysis
# ------------------------------------------------------------------
# This script downloads LIGO H1 and L1 strain data for GW150914,
# applies a bandpass filter, and visualizes the event in both
# the time and time-frequency (Q-transform) domains.

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
# GW150914 event GPS time
MERGER_GPS = 1126259462
# Data interval (±2048s)
START_TIME = MERGER_GPS - 2048
END_TIME = MERGER_GPS + 2048
# Detectors to analyze
DETECTORS = ['H1', 'L1']
# Bandpass filter range
BANDPASS_LOW = 35
BANDPASS_HIGH = 350
# Time-domain plot window (±0.2s)
TD_WINDOW = 0.2  # seconds
# Q-transform spectrogram window (±0.5s)
Q_WINDOW = 0.5  # seconds

# =========================
# 1. Data Download
# =========================
print("="*60)
print(f"STEP 1: Downloading strain data from GWOSC for H1 and L1, GPS {START_TIME} to {END_TIME} (±2048s around {MERGER_GPS})...")
strain_data = {}

for det in DETECTORS:
    try:
        print(f"  Fetching data for {det}...")
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
print(f"STEP 3: Plotting time-domain strain data (±{TD_WINDOW}s around merger)...")

plot_start = MERGER_GPS - TD_WINDOW
plot_end = MERGER_GPS + TD_WINDOW

for det in DETECTORS:
    ts = filtered_strain_data.get(det)
    if ts is None:
        print(f"  WARNING: No filtered data for {det}; skipping plot.")
        continue
    try:
        ts_zoom = ts.crop(plot_start, plot_end)
        times = ts_zoom.times.value - MERGER_GPS  # relative to merger
        strain = ts_zoom.value

        plt.figure(figsize=(10, 4))
        plt.plot(times, strain, label=f'{det} strain')
        plt.title(f"{det} Strain around GW150914 Merger (GPS {MERGER_GPS})")
        plt.xlabel("Time (s) relative to merger")
        plt.ylabel("Strain")
        plt.xlim(-TD_WINDOW, TD_WINDOW)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"    Time-domain plot generated for {det}.")
    except Exception as e:
        print(f"    ERROR: Failed to plot time-domain data for {det}: {e}")
        traceback.print_exc()

print("STEP 3 complete.\n")

# =========================
# 4. Q-transform Spectrograms
# =========================
print("="*60)
print(f"STEP 4: Generating Q-transform spectrograms (±{Q_WINDOW}s around merger)...")

q_plot_start = MERGER_GPS - Q_WINDOW
q_plot_end = MERGER_GPS + Q_WINDOW

for det in DETECTORS:
    ts = filtered_strain_data.get(det)
    if ts is None:
        print(f"  WARNING: No filtered data for {det}; skipping Q-transform.")
        continue
    try:
        ts_zoom = ts.crop(q_plot_start, q_plot_end)
        # Ensure the Q-transform window is not longer than the data segment
        segment_duration = ts_zoom.duration.value
        # Set a Q-transform window that is at most half the segment length
        # (GWPy default is 1s, but our segment is only 1s)
        qwindow = min(0.2, segment_duration / 2.0)
        if qwindow < 0.01:
            print(f"    WARNING: Data segment for {det} is too short for Q-transform. Skipping.")
            continue
        print(f"  Computing Q-transform for {det} with qwindow={qwindow:.3f}s...")
        q = ts_zoom.q_transform(outseg=(q_plot_start, q_plot_end), qrange=(8, 64), frange=(BANDPASS_LOW, BANDPASS_HIGH), window=qwindow)
        print(f"  Plotting Q-transform spectrogram for {det}...")
        fig = q.plot(figsize=(10, 5), vmin=1e-24, vmax=1e-21)
        ax = fig.gca()
        ax.set_title(f"{det} Q-transform Spectrogram around GW150914 (GPS {MERGER_GPS})")
        ax.set_xlabel("Time (s) relative to merger")
        ax.set_ylabel("Frequency (Hz)")
        # Set x-axis to be relative to merger time
        ax.set_xlim(q_plot_start, q_plot_end)
        ticks = ax.get_xticks()
        ax.set_xticklabels([f"{tick - MERGER_GPS:.2f}" for tick in ticks])
        plt.tight_layout()
        plt.show()
        print(f"    Q-transform spectrogram generated for {det}.")
    except Exception as e:
        print(f"    ERROR: Failed to generate Q-transform for {det}: {e}")
        traceback.print_exc()

print("STEP 4 complete.\n")
print("="*60)
print("All analysis steps complete.")