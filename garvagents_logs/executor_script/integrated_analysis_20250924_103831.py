# --- Imports ---
import sys
import numpy as np
import matplotlib.pyplot as plt
from pycbc.catalog import Merger

# --- Utility Functions ---
def print_progress(msg):
    print(f"[INFO] {msg}")

def save_numpy_array(filename, array):
    try:
        np.save(filename, array)
        print_progress(f"Saved array to {filename}.npy")
    except Exception as e:
        print(f"[WARNING] Could not save {filename}: {e}", file=sys.stderr)

# --- Task 1: Download the strain data for GW150914 ---
print_progress("Fetching GW150914 event information from PyCBC catalog...")
strain_H1 = None
strain_L1 = None
gps_time = None

try:
    gw_event = Merger("GW150914")
    gps_time = gw_event.time
    print_progress(f"GW150914 GPS time: {gps_time}")

    # Define a 32-second window centered on the event
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

# --- Task 2: Plot the strain data for GW150914 ---
try:
    if strain_H1 is None or strain_L1 is None:
        raise ValueError("Strain data for H1 or L1 not found. Please run the data loading step first.")

    print_progress("Preparing data for strain plot...")

    # Find overlapping time region
    start_time = max(strain_H1.start_time, strain_L1.start_time)
    end_time = min(strain_H1.end_time, strain_L1.end_time)

    # Build time arrays for the overlapping region
    mask_H1 = (strain_H1.sample_times >= start_time) & (strain_H1.sample_times <= end_time)
    mask_L1 = (strain_L1.sample_times >= start_time) & (strain_L1.sample_times <= end_time)

    times_H1 = strain_H1.sample_times[mask_H1]
    times_L1 = strain_L1.sample_times[mask_L1]

    # For plotting, ensure the time arrays are the same length and aligned
    min_len = min(len(times_H1), len(times_L1))
    times = np.array(times_H1[:min_len]) - float(start_time)  # Relative to start_time

    data_H1 = strain_H1.numpy()[mask_H1][:min_len]
    data_L1 = strain_L1.numpy()[mask_L1][:min_len]

    print_progress("Plotting raw strain data for H1 and L1...")

    plt.figure(figsize=(12, 6))
    plt.plot(times, data_H1, label='H1 (Hanford)', color='C0', alpha=0.8)
    plt.plot(times, data_L1, label='L1 (Livingston)', color='C1', alpha=0.8)
    plt.xlabel('Time (s) relative to segment start')
    plt.ylabel('Strain')
    plt.title('GW150914 Strain Data: H1 and L1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GW150914_raw_strain.png")
    print_progress("Plot saved as GW150914_raw_strain.png")
    plt.show()

    print_progress("Strain data plot displayed successfully.")

except Exception as e:
    print(f"[ERROR] Visualization failed: {e}", file=sys.stderr)
    sys.exit(1)