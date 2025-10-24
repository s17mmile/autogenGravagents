# =========================
# GW150914 Strain Analysis
# =========================

# ---- Imports ----
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pycbc.catalog import Merger

# ---- Section 1: Download GW150914 Strain Data ----
print("\n[1/3] Starting GW150914 strain data download for H1 and L1 detectors...")

strain_H1 = None
strain_L1 = None
times_H1 = None
times_L1 = None
gps_time = None
sample_rate = 4096  # Hz
duration = 32       # seconds

try:
    print("Fetching GW150914 event information from PyCBC catalog...")
    gw_event = Merger("GW150914")
    gps_time = gw_event.time
    print(f"GW150914 GPS time: {gps_time}")

    # Calculate start and end times
    start_time = int(gps_time) - duration // 2
    end_time = start_time + duration

    print(f"Downloading H1 strain data from {start_time} to {end_time} (GPS)...")
    strain_H1 = gw_event.strain('H1', sample_rate=sample_rate, f_low=20, 
                                start_time=start_time, end_time=end_time)
    times_H1 = strain_H1.sample_times
    print("H1 strain data downloaded successfully.")

    print(f"Downloading L1 strain data from {start_time} to {end_time} (GPS)...")
    strain_L1 = gw_event.strain('L1', sample_rate=sample_rate, f_low=20, 
                                start_time=start_time, end_time=end_time)
    times_L1 = strain_L1.sample_times
    print("L1 strain data downloaded successfully.")

    # Save raw data for reproducibility
    np.save("strain_H1_raw.npy", np.array(strain_H1))
    np.save("strain_L1_raw.npy", np.array(strain_L1))
    np.save("times_H1.npy", np.array(times_H1))
    np.save("times_L1.npy", np.array(times_L1))
    print("Raw strain data saved to disk.")

except Exception as e:
    print(f"Error occurred while downloading strain data: {e}")
    sys.exit(1)

# ---- Section 2: Bandpass Filtering (35-350 Hz) ----
print("\n[2/3] Starting bandpass filtering of GW150914 strain data (35-350 Hz)...")

lowcut = 35.0
highcut = 350.0
order = 4  # 4th order Butterworth filter

def bandpass_filter(strain, sample_rate, lowcut, highcut, order=4):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    try:
        filtered = filtfilt(b, a, strain)
        return filtered
    except Exception as e:
        print(f"Error during filtering: {e}")
        return None

try:
    # Convert PyCBC TimeSeries to numpy arrays if needed
    if hasattr(strain_H1, 'sample_rate'):
        sample_rate = float(strain_H1.sample_rate)
        strain_H1_data = np.array(strain_H1)
        strain_L1_data = np.array(strain_L1)
    else:
        strain_H1_data = strain_H1
        strain_L1_data = strain_L1

    print("Applying bandpass filter to H1 data...")
    strain_H1_filtered = bandpass_filter(strain_H1_data, sample_rate, lowcut, highcut, order)
    if strain_H1_filtered is None:
        raise RuntimeError("Bandpass filtering failed for H1.")

    print("Applying bandpass filter to L1 data...")
    strain_L1_filtered = bandpass_filter(strain_L1_data, sample_rate, lowcut, highcut, order)
    if strain_L1_filtered is None:
        raise RuntimeError("Bandpass filtering failed for L1.")

    # Save filtered data
    np.save("strain_H1_filtered.npy", strain_H1_filtered)
    np.save("strain_L1_filtered.npy", strain_L1_filtered)
    print("Filtered strain data saved to disk.")

except Exception as e:
    print(f"Error during bandpass filtering: {e}")
    sys.exit(1)

# ---- Section 3: Visualization ----
print("\n[3/3] Starting visualization of filtered strain data for GW150914...")

try:
    # Convert times to seconds relative to event
    t_H1 = np.array(times_H1) - gps_time
    t_L1 = np.array(times_L1) - gps_time

    # Choose a window around the event (±0.2 seconds)
    window = 0.2  # seconds
    idx_H1 = np.where((t_H1 >= -window) & (t_H1 <= window))
    idx_L1 = np.where((t_L1 >= -window) & (t_L1 <= window))

    plt.figure(figsize=(10, 6))
    plt.plot(t_H1[idx_H1], strain_H1_filtered[idx_H1], label='H1 (Hanford)', color='C0')
    plt.plot(t_L1[idx_L1], strain_L1_filtered[idx_L1], label='L1 (Livingston)', color='C1')
    plt.axvline(0, color='k', linestyle='--', label='GW150914 Event Time')
    plt.title('GW150914 Filtered Strain Time Series (35-350 Hz)')
    plt.xlabel('Time (s) relative to GW150914')
    plt.ylabel('Strain')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GW150914_filtered_strain_plot.png", dpi=150)
    plt.show()
    print("Plot generated and saved as 'GW150914_filtered_strain_plot.png'.")

except Exception as e:
    print(f"Error during visualization: {e}")
    sys.exit(1)

print("\nWorkflow completed successfully.")