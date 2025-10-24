# --- Imports ---
import sys
import numpy as np
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.types import TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from scipy.signal import butter, filtfilt

# --- Utility Functions ---
def print_progress(msg):
    print(f"[INFO] {msg}")

def save_numpy_array(filename, array):
    try:
        np.save(filename, array)
        print_progress(f"Saved array to {filename}.npy")
    except Exception as e:
        print(f"[WARNING] Could not save {filename}: {e}", file=sys.stderr)

# --- Task 1: Download H1 and L1 strain data for GW150914 ---
print_progress("Fetching GW150914 event information from PyCBC catalog...")
strain_H1 = None
strain_L1 = None
gps_time = None
try:
    gw_event = Merger("GW150914")
    gps_time = gw_event.time
    print_progress(f"GW150914 GPS time: {gps_time}")

    duration = 32  # seconds
    start = int(gps_time) - duration // 2
    end = start + duration

    print_progress("Downloading H1 strain data...")
    strain_H1 = gw_event.strain('H1', start, end)
    print_progress("H1 strain data downloaded successfully.")

    print_progress("Downloading L1 strain data...")
    strain_L1 = gw_event.strain('L1', start, end)
    print_progress("L1 strain data downloaded successfully.")

    # Save raw data for reproducibility
    save_numpy_array("strain_H1_raw", strain_H1.numpy())
    save_numpy_array("strain_L1_raw", strain_L1.numpy())

except Exception as e:
    print(f"[ERROR] Failed to download or load strain data: {e}", file=sys.stderr)
    sys.exit(1)

# --- Task 2: Whiten and filter the strain data ---
def whiten_strain(strain, seglen=4, avg_method='median', psd_len=4):
    print_progress("Estimating PSD for whitening...")
    psd = strain.psd(seglen * strain.sample_rate, avg_method=avg_method)
    psd = interpolate(psd, strain.delta_f)
    psd = inverse_spectrum_truncation(psd, int(strain.sample_rate), low_frequency_cutoff=15)
    print_progress("Whitening strain data...")
    whitened = strain.whiten(4, 2, psd=psd)
    return whitened

def butter_band_filter(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    print_progress(f"Applying bandpass filter: {lowcut} Hz - {highcut} Hz")
    filtered = filtfilt(b, a, data)
    return filtered

whitened_H1 = None
whitened_L1 = None
processed_H1 = None
processed_L1 = None

try:
    print_progress("Processing H1 data...")
    whitened_H1 = whiten_strain(strain_H1)
    print_progress("Processing L1 data...")
    whitened_L1 = whiten_strain(strain_L1)

    fs_H1 = whitened_H1.sample_rate
    fs_L1 = whitened_L1.sample_rate

    processed_H1 = TimeSeries(
        butter_band_filter(whitened_H1.numpy(), fs_H1, 30, 250),
        delta_t=whitened_H1.delta_t,
        epoch=whitened_H1.start_time
    )
    print_progress("H1 data filtered.")

    processed_L1 = TimeSeries(
        butter_band_filter(whitened_L1.numpy(), fs_L1, 30, 250),
        delta_t=whitened_L1.delta_t,
        epoch=whitened_L1.start_time
    )
    print_progress("L1 data filtered.")

    # Save processed data
    save_numpy_array("processed_H1", processed_H1.numpy())
    save_numpy_array("processed_L1", processed_L1.numpy())

except Exception as e:
    print(f"[ERROR] Processing failed: {e}", file=sys.stderr)
    sys.exit(1)

# --- Task 3: Plot the whitened and filtered H1 and L1 strain data ---
try:
    if processed_H1 is None or processed_L1 is None:
        raise ValueError("Processed H1 or L1 data not found. Please run the processing step first.")

    print_progress("Preparing data for plotting...")

    # Find overlapping time region
    start_time = max(processed_H1.start_time, processed_L1.start_time)
    end_time = min(processed_H1.end_time, processed_L1.end_time)

    mask_H1 = (processed_H1.sample_times >= start_time) & (processed_H1.sample_times <= end_time)
    mask_L1 = (processed_L1.sample_times >= start_time) & (processed_L1.sample_times <= end_time)

    times_H1 = processed_H1.sample_times[mask_H1]
    times_L1 = processed_L1.sample_times[mask_L1]

    min_len = min(len(times_H1), len(times_L1))
    times = np.array(times_H1[:min_len]) - float(start_time)

    data_H1 = processed_H1.numpy()[mask_H1][:min_len]
    data_L1 = processed_L1.numpy()[mask_L1][:min_len]

    print_progress("Plotting whitened and filtered strain data...")

    plt.figure(figsize=(12, 6))
    plt.plot(times, data_H1, label='H1 (Hanford)', color='C0', alpha=0.8)
    plt.plot(times, data_L1, label='L1 (Livingston)', color='C1', alpha=0.8)
    plt.xlabel('Time (s) relative to segment start')
    plt.ylabel('Whitened Strain')
    plt.title('Whitened and Filtered Strain Data: GW150914 (H1 & L1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GW150914_whitened_filtered_strain.png")
    print_progress("Plot saved as GW150914_whitened_filtered_strain.png")
    plt.show()

    print_progress("Plot displayed successfully.")

except Exception as e:
    print(f"[ERROR] Visualization failed: {e}", file=sys.stderr)
    sys.exit(1)