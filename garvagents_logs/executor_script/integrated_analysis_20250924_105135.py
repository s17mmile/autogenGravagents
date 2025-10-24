# GW150914 Gravitational Wave Data Analysis Script
# ------------------------------------------------
# This script downloads, filters, and visualizes strain data for GW150914 using GWpy and matplotlib.

import os
import sys
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# -------------------------------
# PARAMETERS AND OUTPUT DIRECTORY
# -------------------------------
EVENT_GPS = 1126259462
DELTA = 2048  # seconds before and after event for data download
LOW_FREQ = 35
HIGH_FREQ = 350
PLOT_WINDOW = 0.2  # seconds for time-domain plot
SPEC_WINDOW = 2    # seconds for spectrogram
Q_LOW_FREQ = 30
Q_HIGH_FREQ = 400
OUTPUT_DIR = "gw150914_analysis_outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. DATA LOADING
# -------------------------------
print("="*60)
print("Step 1: Downloading H1 and L1 strain data from GWOSC")
gps_start = EVENT_GPS - DELTA
gps_end = EVENT_GPS + DELTA

h1 = None
l1 = None

try:
    print(f"Fetching H1 strain data ({gps_start} to {gps_end} GPS)...")
    h1 = TimeSeries.fetch_open_data('H1', gps_start, gps_end, cache=True)
    print("H1 strain data downloaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to download H1 data: {e}")
    sys.exit(1)

try:
    print(f"Fetching L1 strain data ({gps_start} to {gps_end} GPS)...")
    l1 = TimeSeries.fetch_open_data('L1', gps_start, gps_end, cache=True)
    print("L1 strain data downloaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to download L1 data: {e}")
    sys.exit(1)

# Optionally save raw data to disk
try:
    h1.write(os.path.join(OUTPUT_DIR, "H1_raw.gwf"), format='gwf')
    l1.write(os.path.join(OUTPUT_DIR, "L1_raw.gwf"), format='gwf')
    print("Raw strain data saved to disk.")
except Exception as e:
    print(f"WARNING: Could not save raw data: {e}")

# -------------------------------
# 2. FILTERING
# -------------------------------
print("="*60)
print(f"Step 2: Applying Butterworth bandpass filter ({LOW_FREQ}-{HIGH_FREQ} Hz)")

h1_filtered = None
l1_filtered = None

try:
    print("Filtering H1 strain data...")
    h1_filtered = h1.bandpass(LOW_FREQ, HIGH_FREQ, filtfilt=True)
    print("H1 strain data filtered.")
except Exception as e:
    print(f"ERROR: Failed to filter H1 data: {e}")
    sys.exit(1)

try:
    print("Filtering L1 strain data...")
    l1_filtered = l1.bandpass(LOW_FREQ, HIGH_FREQ, filtfilt=True)
    print("L1 strain data filtered.")
except Exception as e:
    print(f"ERROR: Failed to filter L1 data: {e}")
    sys.exit(1)

# Optionally save filtered data to disk
try:
    h1_filtered.write(os.path.join(OUTPUT_DIR, "H1_filtered.gwf"), format='gwf')
    l1_filtered.write(os.path.join(OUTPUT_DIR, "L1_filtered.gwf"), format='gwf')
    print("Filtered strain data saved to disk.")
except Exception as e:
    print(f"WARNING: Could not save filtered data: {e}")

# -------------------------------
# 3. TIME-DOMAIN PLOT
# -------------------------------
print("="*60)
print(f"Step 3: Creating time-domain plots (±{PLOT_WINDOW} s around event)")

try:
    # Crop filtered data to ±PLOT_WINDOW seconds around the event
    h1_zoom = h1_filtered.crop(EVENT_GPS - PLOT_WINDOW, EVENT_GPS + PLOT_WINDOW)
    l1_zoom = l1_filtered.crop(EVENT_GPS - PLOT_WINDOW, EVENT_GPS + PLOT_WINDOW)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(h1_zoom.times.value - EVENT_GPS, h1_zoom.value, label='H1', color='C0')
    plt.plot(l1_zoom.times.value - EVENT_GPS, l1_zoom.value, label='L1', color='C1')
    plt.xlabel('Time (s) relative to GW150914')
    plt.ylabel('Strain')
    plt.title('GW150914: Filtered Strain Data Around Merger Event')
    plt.legend()
    plt.grid(True)
    plt.xlim(-PLOT_WINDOW, PLOT_WINDOW)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "GW150914_time_domain.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Time-domain plot saved to {plot_path}")
except Exception as e:
    print(f"ERROR: Failed to create time-domain plot: {e}")

# -------------------------------
# 4. SPECTROGRAM (Q-TRANSFORM)
# -------------------------------
print("="*60)
print(f"Step 4: Generating Q-transform spectrograms (±{SPEC_WINDOW} s around event)")

try:
    # Crop filtered data to ±SPEC_WINDOW seconds around the event
    h1_spec = h1_filtered.crop(EVENT_GPS - SPEC_WINDOW, EVENT_GPS + SPEC_WINDOW)
    l1_spec = l1_filtered.crop(EVENT_GPS - SPEC_WINDOW, EVENT_GPS + SPEC_WINDOW)

    # Q-transform for H1
    print("Computing Q-transform for H1...")
    h1_q = h1_spec.q_transform(frange=(Q_LOW_FREQ, Q_HIGH_FREQ))
    print("Q-transform for H1 computed.")

    # Q-transform for L1
    print("Computing Q-transform for L1...")
    l1_q = l1_spec.q_transform(frange=(Q_LOW_FREQ, Q_HIGH_FREQ))
    print("Q-transform for L1 computed.")

    # Plot H1 spectrogram
    print("Plotting H1 Q-transform spectrogram...")
    fig1 = h1_q.plot()
    ax1 = fig1.gca()
    ax1.set_title('GW150914 H1 Q-transform Spectrogram')
    ax1.set_xlim(EVENT_GPS - SPEC_WINDOW, EVENT_GPS + SPEC_WINDOW)
    ax1.set_ylim(Q_LOW_FREQ, Q_HIGH_FREQ)
    plt.tight_layout()
    h1_spec_path = os.path.join(OUTPUT_DIR, "GW150914_H1_spectrogram.png")
    plt.savefig(h1_spec_path)
    plt.show()
    print(f"H1 spectrogram saved to {h1_spec_path}")

    # Plot L1 spectrogram
    print("Plotting L1 Q-transform spectrogram...")
    fig2 = l1_q.plot()
    ax2 = fig2.gca()
    ax2.set_title('GW150914 L1 Q-transform Spectrogram')
    ax2.set_xlim(EVENT_GPS - SPEC_WINDOW, EVENT_GPS + SPEC_WINDOW)
    ax2.set_ylim(Q_LOW_FREQ, Q_HIGH_FREQ)
    plt.tight_layout()
    l1_spec_path = os.path.join(OUTPUT_DIR, "GW150914_L1_spectrogram.png")
    plt.savefig(l1_spec_path)
    plt.show()
    print(f"L1 spectrogram saved to {l1_spec_path}")

    print("Spectrograms created successfully.")
except Exception as e:
    print(f"ERROR: Failed to generate or plot spectrograms: {e}")

print("="*60)
print("Analysis complete. All outputs saved in:", OUTPUT_DIR)