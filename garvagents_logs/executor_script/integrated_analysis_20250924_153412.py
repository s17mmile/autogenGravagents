# GW150914 Strain Data Download, Filtering, and Analysis
# ------------------------------------------------------
# This script downloads LIGO H1 and L1 strain data for GW150914,
# applies a bandpass filter, and visualizes the event in both
# the time and time-frequency domains.

# =========================
# Imports and Configuration
# =========================
import sys
import traceback
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Parameters
# =========================
# GW150914 event GPS time
MERGER_GPS = 1126259462
# Data interval (±2048s)
DELTA = 2048
START = MERGER_GPS - DELTA
END = MERGER_GPS + DELTA
# Detectors to analyze
DETECTORS = ['H1', 'L1']
# Bandpass filter range
BANDPASS_LOW = 35
BANDPASS_HIGH = 350
# Time-domain plot window (±0.2s)
TD_WINDOW = 0.2  # seconds
# Spectrogram window (±0.5s)
SPEC_WINDOW = 0.5  # seconds

# =========================
# 1. Data Download
# =========================
print("="*60)
print(f"STEP 1: Downloading strain data from GWOSC for H1 and L1, GPS {START} to {END} (±{DELTA}s around {MERGER_GPS})...")
strain_data = {}

for det in DETECTORS:
    try:
        print(f"  Fetching data for {det}...")
        ts = TimeSeries.fetch_open_data(det, START, END, cache=True)
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

plt.figure(figsize=(10, 6))
for det in DETECTORS:
    ts = filtered_strain_data.get(det)
    if ts is None:
        print(f"  WARNING: No filtered data for {det}; skipping plot.")
        continue
    try:
        ts_zoom = ts.crop(plot_start, plot_end)
        times = ts_zoom.times.value - MERGER_GPS
        plt.plot(times, ts_zoom.value, label=det)
        print(f"    Plotted {det}.")
    except Exception as e:
        print(f"    ERROR: Failed to plot {det}: {e}")
        traceback.print_exc()

plt.xlabel('Time (s) relative to merger')
plt.ylabel('Strain')
plt.title(f'Filtered Strain Data Around Merger (GPS {MERGER_GPS} ± {TD_WINDOW} s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("STEP 3 complete.\n")

# =========================
# 4. Time-Frequency Spectrograms
# =========================
print("="*60)
print(f"STEP 4: Generating time-frequency spectrograms (±{SPEC_WINDOW}s around merger)...")

spec_start = MERGER_GPS - SPEC_WINDOW
spec_end = MERGER_GPS + SPEC_WINDOW

for det in DETECTORS:
    ts = filtered_strain_data.get(det)
    if ts is None:
        print(f"  WARNING: No filtered data for {det}; skipping spectrogram.")
        continue
    try:
        ts_zoom = ts.crop(spec_start, spec_end)
        data = ts_zoom.value
        times = ts_zoom.times.value - MERGER_GPS  # relative to merger
        fs = ts_zoom.sample_rate.value

        plt.figure(figsize=(10, 5))
        NFFT = int(fs * 0.0625)  # window length ~62.5 ms
        noverlap = int(NFFT * 0.9)
        plt.specgram(data, NFFT=NFFT, Fs=fs, noverlap=noverlap, cmap='viridis', scale='dB')
        plt.title(f"{det} Time-Frequency Spectrogram (GW150914, GPS {MERGER_GPS})")
        plt.xlabel("Time (s) relative to merger")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label='dB')
        plt.xlim(-SPEC_WINDOW, SPEC_WINDOW)
        plt.ylim(20, 400)
        plt.tight_layout()
        plt.show()
        print(f"    Spectrogram plotted for {det}.")
    except Exception as e:
        print(f"    ERROR: Failed to generate spectrogram for {det}: {e}")
        traceback.print_exc()

print("STEP 4 complete.\n")
print("="*60)
print("All analysis steps complete.")