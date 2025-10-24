# GW170817 Gravitational Wave Event: Strain and Spectrogram Visualization

import os
import sys
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# -------------------------------
# PARAMETERS AND OUTPUT DIRECTORY
# -------------------------------
EVENT_GPS = 1187008882
DATA_WINDOW = 10      # seconds before and after event for data download
FILTER_LOW = 30       # Hz
FILTER_HIGH = 500     # Hz
PLOT_WINDOW = 4       # seconds before and after event for plotting/spectrogram
OUTPUT_DIR = "gw170817_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. DATA LOADING
# -------------------------------
print("="*60)
print(f"Step 1: Downloading H1 and L1 strain data for GW170817 (GPS {EVENT_GPS}) ±{DATA_WINDOW} seconds")

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
print(f"Step 3: Plotting filtered strain data for GW170817 (±{PLOT_WINDOW} seconds)")

try:
    print(f"Cropping filtered data to ±{PLOT_WINDOW} seconds around GW170817 (GPS {EVENT_GPS})...")
    h1_plot = h1_filtered.crop(EVENT_GPS - PLOT_WINDOW, EVENT_GPS + PLOT_WINDOW)
    l1_plot = l1_filtered.crop(EVENT_GPS - PLOT_WINDOW, EVENT_GPS + PLOT_WINDOW)

    print("Plotting filtered strain data for H1 and L1...")
    plt.figure(figsize=(12, 6))
    plt.plot(h1_plot.times.value - EVENT_GPS, h1_plot.value, label='H1', color='C0')
    plt.plot(l1_plot.times.value - EVENT_GPS, l1_plot.value, label='L1', color='C1')
    plt.axvline(0, color='k', linestyle='--', label='GW170817 Event Time')
    plt.xlabel('Time (s) relative to GW170817')
    plt.ylabel('Strain')
    plt.title('GW170817: Filtered Strain Data (Time Domain)')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "GW170817_waveform.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Waveform plot saved to {plot_path}")
except Exception as e:
    print(f"ERROR: Failed to plot filtered strain data: {e}")

# -------------------------------
# 4. SPECTROGRAM (Q-TRANSFORM)
# -------------------------------
print("="*60)
print(f"Step 4: Generating Q-transform spectrograms for GW170817 (±{PLOT_WINDOW} seconds)")

try:
    print(f"Cropping filtered data to ±{PLOT_WINDOW} seconds around GW170817 (GPS {EVENT_GPS}) for spectrogram...")
    h1_spec = h1_filtered.crop(EVENT_GPS - PLOT_WINDOW, EVENT_GPS + PLOT_WINDOW)
    l1_spec = l1_filtered.crop(EVENT_GPS - PLOT_WINDOW, EVENT_GPS + PLOT_WINDOW)

    print("Generating Q-transform spectrogram for H1...")
    h1_q = h1_spec.q_transform(outseg=(EVENT_GPS - PLOT_WINDOW, EVENT_GPS + PLOT_WINDOW))
    print("Generating Q-transform spectrogram for L1...")
    l1_q = l1_spec.q_transform(outseg=(EVENT_GPS - PLOT_WINDOW, EVENT_GPS + PLOT_WINDOW))

    # Plot H1 spectrogram
    print("Plotting H1 Q-transform spectrogram...")
    fig1 = h1_q.plot(figsize=(12, 5))
    ax1 = fig1.gca()
    ax1.set_title('GW170817: H1 Q-transform Spectrogram')
    ax1.axvline(EVENT_GPS, color='w', linestyle='--', label='GW170817 Event Time')
    ax1.legend()
    plt.tight_layout()
    h1_spec_path = os.path.join(OUTPUT_DIR, "GW170817_H1_spectrogram.png")
    plt.savefig(h1_spec_path)
    plt.show()
    print(f"H1 spectrogram saved to {h1_spec_path}")

    # Plot L1 spectrogram
    print("Plotting L1 Q-transform spectrogram...")
    fig2 = l1_q.plot(figsize=(12, 5))
    ax2 = fig2.gca()
    ax2.set_title('GW170817: L1 Q-transform Spectrogram')
    ax2.axvline(EVENT_GPS, color='w', linestyle='--', label='GW170817 Event Time')
    ax2.legend()
    plt.tight_layout()
    l1_spec_path = os.path.join(OUTPUT_DIR, "GW170817_L1_spectrogram.png")
    plt.savefig(l1_spec_path)
    plt.show()
    print(f"L1 spectrogram saved to {l1_spec_path}")

    print("Spectrograms created successfully.")
except Exception as e:
    print(f"ERROR: Failed to generate or plot spectrograms: {e}")

print("="*60)
print("Analysis complete. All outputs saved in:", OUTPUT_DIR)