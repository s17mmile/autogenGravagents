# ==========================
# GW150914 Q-Transform Spectrogram Analysis
# ==========================

# ---- Imports ----
import sys
import numpy as np
import matplotlib.pyplot as plt

from pycbc.catalog import Merger
from pycbc.types import TimeSeries
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

# ---- Constants ----
EVENT_NAME = "GW150914"
EVENT_GPS = 1126259462
DURATION = 4  # seconds
START_TIME = EVENT_GPS - DURATION // 2
END_TIME = EVENT_GPS + DURATION // 2
LOW_FREQ = 35.0
HIGH_FREQ = 350.0

# ---- Utility Functions ----
def save_timeseries(ts, filename):
    """Save PyCBC TimeSeries to numpy file."""
    try:
        np.savez(filename, times=ts.sample_times, data=ts.numpy())
        print(f"Saved TimeSeries to {filename}.npz")
    except Exception as e:
        print(f"Warning: Could not save TimeSeries to {filename}: {e}")

def save_qspec(qspec, filename):
    """Save Q spectrogram as a numpy array (magnitude only)."""
    try:
        np.savez(filename, times=qspec.xindex.value, freqs=qspec.yindex.value, magnitude=qspec.value)
        print(f"Saved Q spectrogram to {filename}.npz")
    except Exception as e:
        print(f"Warning: Could not save Q spectrogram to {filename}: {e}")

# ==========================
# 1. Download GW150914 Strain Data
# ==========================
print("\n=== [1/4] Downloading GW150914 strain data ===")
strain_H1 = None
strain_L1 = None

try:
    print("Fetching GW150914 event metadata from PyCBC catalog...")
    merger = Merger(EVENT_NAME)
    print(f"Event time (GPS): {merger.time}")

    print("Downloading Hanford (H1) strain data...")
    strain_H1 = merger.strain('H1', start_time=START_TIME, end_time=END_TIME)
    print(f"Hanford (H1) strain data loaded: {strain_H1}")

    print("Downloading Livingston (L1) strain data...")
    strain_L1 = merger.strain('L1', start_time=START_TIME, end_time=END_TIME)
    print(f"Livingston (L1) strain data loaded: {strain_L1}")

    # Save raw data
    save_timeseries(strain_H1, "GW150914_H1_raw")
    save_timeseries(strain_L1, "GW150914_L1_raw")

except Exception as e:
    print(f"Error occurred while downloading strain data: {e}", file=sys.stderr)
    sys.exit(1)

# ==========================
# 2. Preprocess: Bandpass Filter
# ==========================
print("\n=== [2/4] Bandpass filtering strain data (35-350 Hz) ===")
filtered_strain_H1 = None
filtered_strain_L1 = None

try:
    print("Applying bandpass filter to Hanford (H1)...")
    filtered_strain_H1 = strain_H1.bandpass(LOW_FREQ, HIGH_FREQ, filtfilt=True)
    print("Bandpass filtering complete for Hanford (H1).")

    print("Applying bandpass filter to Livingston (L1)...")
    filtered_strain_L1 = strain_L1.bandpass(LOW_FREQ, HIGH_FREQ, filtfilt=True)
    print("Bandpass filtering complete for Livingston (L1).")

    # Save filtered data
    save_timeseries(filtered_strain_H1, "GW150914_H1_filtered")
    save_timeseries(filtered_strain_L1, "GW150914_L1_filtered")

except Exception as e:
    print(f"Error during bandpass filtering: {e}", file=sys.stderr)
    sys.exit(1)

# ==========================
# 3. Q-Transform (Q Spectrogram)
# ==========================
print("\n=== [3/4] Computing Q-transform spectrograms ===")
qspec_H1 = None
qspec_L1 = None

try:
    print("Converting filtered Hanford (H1) strain data to GWpy TimeSeries...")
    gwpy_strain_H1 = GWpyTimeSeries(filtered_strain_H1.numpy(),
                                    sample_rate=filtered_strain_H1.sample_rate,
                                    t0=filtered_strain_H1.start_time)
    print("Conversion complete for Hanford (H1).")

    print("Converting filtered Livingston (L1) strain data to GWpy TimeSeries...")
    gwpy_strain_L1 = GWpyTimeSeries(filtered_strain_L1.numpy(),
                                    sample_rate=filtered_strain_L1.sample_rate,
                                    t0=filtered_strain_L1.start_time)
    print("Conversion complete for Livingston (L1).")

    # Compute Q-transform (Q spectrogram) for Hanford
    print("Computing Q-transform for Hanford (H1)...")
    qspec_H1 = gwpy_strain_H1.q_transform(
        outseg=(gwpy_strain_H1.t0.value, gwpy_strain_H1.t0.value + gwpy_strain_H1.duration.value)
    )
    print("Q-transform complete for Hanford (H1).")

    # Compute Q-transform (Q spectrogram) for Livingston
    print("Computing Q-transform for Livingston (L1)...")
    qspec_L1 = gwpy_strain_L1.q_transform(
        outseg=(gwpy_strain_L1.t0.value, gwpy_strain_L1.t0.value + gwpy_strain_L1.duration.value)
    )
    print("Q-transform complete for Livingston (L1).")

    # Save Q spectrograms
    save_qspec(qspec_H1, "GW150914_H1_qspec")
    save_qspec(qspec_L1, "GW150914_L1_qspec")

except Exception as e:
    print(f"Error during Q-transform computation: {e}", file=sys.stderr)
    sys.exit(1)

# ==========================
# 4. Visualization: Plot Q Spectrograms
# ==========================
print("\n=== [4/4] Plotting Q-transform spectrograms ===")
try:
    # Hanford (H1)
    print("Plotting Q spectrogram for Hanford (H1)...")
    fig_H1 = qspec_H1.plot(figsize=(10, 6), vmin=1e-24, vmax=1e-21)
    ax_H1 = fig_H1.gca()
    ax_H1.axvline(EVENT_GPS, color='red', linestyle='--', label='GW150914')
    ax_H1.legend()
    ax_H1.set_title("Hanford (H1) Q-transform Spectrogram\nGW150914")
    fig_H1.savefig("GW150914_H1_qspec.png")
    print("Hanford (H1) Q spectrogram saved as GW150914_H1_qspec.png.")
    plt.show()

    # Livingston (L1)
    print("Plotting Q spectrogram for Livingston (L1)...")
    fig_L1 = qspec_L1.plot(figsize=(10, 6), vmin=1e-24, vmax=1e-21)
    ax_L1 = fig_L1.gca()
    ax_L1.axvline(EVENT_GPS, color='red', linestyle='--', label='GW150914')
    ax_L1.legend()
    ax_L1.set_title("Livingston (L1) Q-transform Spectrogram\nGW150914")
    fig_L1.savefig("GW150914_L1_qspec.png")
    print("Livingston (L1) Q spectrogram saved as GW150914_L1_qspec.png.")
    plt.show()

    print("Q spectrogram plotting complete.")

except Exception as e:
    print(f"Error during Q spectrogram plotting: {e}", file=sys.stderr)
    sys.exit(1)

print("\n=== Analysis complete. All results saved in current directory. ===")