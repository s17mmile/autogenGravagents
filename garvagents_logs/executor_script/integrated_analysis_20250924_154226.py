# GW150914 Strain Data Download, Filtering, and Q-transform Analysis
# ------------------------------------------------------------------
# This script downloads LIGO H1 and L1 strain data for GW150914,
# applies a bandpass filter, and visualizes the event in both
# the time and time-frequency (Q-transform) domains.

import sys
import traceback
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np

# For alternative Q-transform
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import highpass, lowpass
from pycbc.events import qtransform

# =========================
# Parameters
# =========================
MERGER_GPS = 1126259462
START_TIME = MERGER_GPS - 2048
END_TIME = MERGER_GPS + 2048
DETECTORS = ['H1', 'L1']
BANDPASS_LOW = 35
BANDPASS_HIGH = 350
TD_WINDOW = 0.2  # seconds
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
# 4. Q-transform Spectrograms (PyCBC alternative)
# =========================
print("="*60)
print(f"STEP 4: Generating Q-transform spectrograms (±{Q_WINDOW}s around merger) using PyCBC...")

q_plot_start = MERGER_GPS - Q_WINDOW
q_plot_end = MERGER_GPS + Q_WINDOW

for det in DETECTORS:
    ts = filtered_strain_data.get(det)
    if ts is None:
        print(f"  WARNING: No filtered data for {det}; skipping Q-transform.")
        continue
    try:
        ts_zoom = ts.crop(q_plot_start, q_plot_end)
        # Convert GWPy TimeSeries to PyCBC TimeSeries
        pycbc_ts = PyCBC_TimeSeries(ts_zoom.value, delta_t=ts_zoom.dt.value, epoch=ts_zoom.t0.value)
        # PyCBC Q-transform
        print(f"  Computing Q-transform for {det} using PyCBC...")
        qgram, times, freqs = qtransform.qtransform(pycbc_ts, logfsteps=100, qrange=(8, 64), frange=(BANDPASS_LOW, BANDPASS_HIGH))
        # Plot
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(times - MERGER_GPS, freqs, np.abs(qgram), shading='auto', cmap='viridis')
        plt.title(f"{det} Q-transform Spectrogram around GW150914 (GPS {MERGER_GPS})")
        plt.xlabel("Time (s) relative to merger")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(BANDPASS_LOW, BANDPASS_HIGH)
        plt.colorbar(label='Q-transform amplitude')
        plt.tight_layout()
        plt.show()
        print(f"    Q-transform spectrogram generated for {det}.")
    except Exception as e:
        print(f"    ERROR: Failed to generate Q-transform for {det}: {e}")
        traceback.print_exc()

print("STEP 4 complete.\n")
print("="*60)
print("All analysis steps complete.")