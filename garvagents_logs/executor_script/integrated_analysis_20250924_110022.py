# GW150914 Gravitational Wave Event Plotting Script

import os
import sys
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# -------------------------------
# PARAMETERS AND OUTPUT DIRECTORY
# -------------------------------
EVENT_GPS = 1126259462
DATA_WINDOW = 5      # seconds before and after event for data download
FILTER_LOW = 35      # Hz
FILTER_HIGH = 350    # Hz
PLOT_WINDOW = 0.2    # seconds before and after event for plotting
OUTPUT_DIR = "gw150914_waveform_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. DATA LOADING
# -------------------------------
print("="*60)
print(f"Step 1: Downloading H1 and L1 strain data for GW150914 (GPS {EVENT_GPS}) ±{DATA_WINDOW} seconds")

h1 = None
l1 = None

try:
    print("Downloading H1 data...")
    h1 = TimeSeries.fetch_open_data('H1', EVENT_GPS - DATA_WINDOW, EVENT_GPS + DATA_WINDOW, cache=True)
    print("H1 data downloaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to download H1 data: {e}")
    sys.exit(1)

try:
    print("Downloading L1 data...")
    l1 = TimeSeries.fetch_open_data('L1', EVENT_GPS - DATA_WINDOW, EVENT_GPS + DATA_WINDOW, cache=True)
    print("L1 data downloaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to download L1 data: {e}")
    sys.exit(1)

# Optionally save raw data
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
print(f"Step 2: Applying {FILTER_LOW}-{FILTER_HIGH} Hz Butterworth bandpass filter")

h1_filtered = None
l1_filtered = None

try:
    print(f"Filtering H1 data...")
    h1_filtered = h1.bandpass(FILTER_LOW, FILTER_HIGH, filtfilt=True)
    print("H1 data filtered successfully.")
except Exception as e:
    print(f"ERROR: Failed to filter H1 data: {e}")
    sys.exit(1)

try:
    print(f"Filtering L1 data...")
    l1_filtered = l1.bandpass(FILTER_LOW, FILTER_HIGH, filtfilt=True)
    print("L1 data filtered successfully.")
except Exception as e:
    print(f"ERROR: Failed to filter L1 data: {e}")
    sys.exit(1)

# Optionally save filtered data
try:
    h1_filtered.write(os.path.join(OUTPUT_DIR, "H1_filtered.gwf"), format='gwf')
    l1_filtered.write(os.path.join(OUTPUT_DIR, "L1_filtered.gwf"), format='gwf')
    print("Filtered strain data saved to disk.")
except Exception as e:
    print(f"WARNING: Could not save filtered data: {e}")

# -------------------------------
# 3. PLOTTING WAVEFORM
# -------------------------------
print("="*60)
print(f"Step 3: Plotting filtered strain data for GW150914 (±{PLOT_WINDOW} seconds)")

try:
    print(f"Cropping filtered data to ±{PLOT_WINDOW} seconds around GW150914 (GPS {EVENT_GPS})...")
    h1_plot = h1_filtered.crop(EVENT_GPS - PLOT_WINDOW, EVENT_GPS + PLOT_WINDOW)
    l1_plot = l1_filtered.crop(EVENT_GPS - PLOT_WINDOW, EVENT_GPS + PLOT_WINDOW)

    print("Plotting filtered strain data for H1 and L1...")
    plt.figure(figsize=(10, 6))
    plt.plot(h1_plot.times.value - EVENT_GPS, h1_plot.value, label='H1', color='C0')
    plt.plot(l1_plot.times.value - EVENT_GPS, l1_plot.value, label='L1', color='C1')
    plt.axvline(0, color='k', linestyle='--', label='GW150914 Event Time')
    plt.xlabel('Time (s) relative to GW150914')
    plt.ylabel('Strain')
    plt.title('GW150914: Filtered Strain Data (Time Domain)')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "GW150914_waveform.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Waveform plot saved to {plot_path}")
except Exception as e:
    print(f"ERROR: Failed to plot filtered strain data: {e}")

print("="*60)
print("Analysis complete. All outputs saved in:", OUTPUT_DIR)