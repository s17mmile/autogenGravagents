# GW150914 H1 and L1 Strain Data: Download, Whiten, Bandpass Filter, and Plot
# ---------------------------------------------------------------------------
# This script downloads the GW150914 strain data for H1 and L1, whitens it,
# applies a 30-250 Hz bandpass filter, and plots both time series for comparison.

import sys
import time

print("Importing required modules...")
try:
    from pycbc.catalog import Merger
    from pycbc.types import TimeSeries
    import numpy as np
    from scipy.signal import butter, filtfilt
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
print("All modules imported successfully.\n")

# =========================
# Task 1: Data Loading
# =========================
print("="*60)
print("TASK 1: Downloading GW150914 H1 and L1 strain data...")
event_name = "GW150914"
detectors = ["H1", "L1"]
strain_data = {}

# Helper: Retry download up to 3 times
def download_strain_with_retries(event, det, max_retries=3, delay=5):
    for attempt in range(1, max_retries+1):
        try:
            print(f"  Attempt {attempt}: Downloading strain data for detector {det}...")
            strain = event.strain(det)
            print(f"    {det}: Duration = {strain.duration} s, Sample rate = {strain.sample_rate} Hz")
            return strain
        except Exception as e:
            print(f"    Error downloading strain data for {det} (attempt {attempt}): {e}")
            if attempt < max_retries:
                print(f"    Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"    Failed to download strain data for {det} after {max_retries} attempts.")
    return None

try:
    event = Merger(event_name)
    print(f"Event '{event_name}' data fetched successfully.")
except Exception as e:
    print(f"Error fetching event data: {e}")
    sys.exit(1)

for det in detectors:
    strain_data[det] = download_strain_with_retries(event, det)

# Assign to variables for next steps
strain_H1 = strain_data["H1"]
strain_L1 = strain_data["L1"]

if strain_H1 is None or strain_L1 is None:
    print("Error: Could not download both H1 and L1 strain data. Exiting.")
    sys.exit(1)
print("TASK 1 complete.\n")

# =========================
# Task 2: Whitening
# =========================
print("="*60)
print("TASK 2: Whitening strain data...")

def whiten_strain(strain, seg_len=4, seg_stride=2):
    try:
        print(f"  Estimating PSD (segment length: {seg_len}s, stride: {seg_stride}s)...")
        # Use the documented API for PSD estimation
        psd = strain.psd(int(seg_len * strain.sample_rate), int(seg_stride * strain.sample_rate))
        print("  PSD estimation complete.")
        print("  Whitening strain data...")
        whitened = strain.whiten(4, 2, psd=psd)
        print("  Whitening complete.")
        return whitened
    except Exception as e:
        print(f"  Error during whitening: {e}")
        return None

print("Processing H1 strain data...")
whitened_strain_H1 = whiten_strain(strain_H1)
if whitened_strain_H1 is None:
    print("Error: Whitening failed for H1. Exiting.")
    sys.exit(1)

print("Processing L1 strain data...")
whitened_strain_L1 = whiten_strain(strain_L1)
if whitened_strain_L1 is None:
    print("Error: Whitening failed for L1. Exiting.")
    sys.exit(1)
print("TASK 2 complete.\n")

# =========================
# Task 3: Bandpass Filtering
# =========================
print("="*60)
print("TASK 3: Applying bandpass filter (30-250 Hz)...")

def bandpass_filter(strain, lowcut=30.0, highcut=250.0, order=4):
    try:
        print(f"  Applying bandpass filter: {lowcut} Hz - {highcut} Hz (order {order})...")
        fs = float(strain.sample_rate)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if not (0 < low < 1 and 0 < high < 1 and low < high):
            raise ValueError("Invalid bandpass filter frequencies.")
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, strain.numpy())
        filtered_strain = TimeSeries(filtered_data, delta_t=strain.delta_t, epoch=strain.start_time)
        print("  Bandpass filtering complete.")
        return filtered_strain
    except Exception as e:
        print(f"  Error during bandpass filtering: {e}")
        return None

print("Filtering H1 whitened strain data...")
bandpassed_strain_H1 = bandpass_filter(whitened_strain_H1)
if bandpassed_strain_H1 is None:
    print("Error: Bandpass filtering failed for H1. Exiting.")
    sys.exit(1)

print("Filtering L1 whitened strain data...")
bandpassed_strain_L1 = bandpass_filter(whitened_strain_L1)
if bandpassed_strain_L1 is None:
    print("Error: Bandpass filtering failed for L1. Exiting.")
    sys.exit(1)
print("TASK 3 complete.\n")

# =========================
# Task 4: Plotting
# =========================
print("="*60)
print("TASK 4: Plotting processed strain data...")

try:
    # Extract time arrays relative to the start time for both detectors
    t_H1 = bandpassed_strain_H1.sample_times - float(bandpassed_strain_H1.start_time)
    t_L1 = bandpassed_strain_L1.sample_times - float(bandpassed_strain_L1.start_time)
    y_H1 = bandpassed_strain_H1.numpy()
    y_L1 = bandpassed_strain_L1.numpy()
    if not (np.isfinite(y_H1).all() and np.isfinite(y_L1).all()):
        raise ValueError("Non-finite values detected in strain data.")
    print("  Data prepared for plotting.")
except Exception as e:
    print(f"  Error preparing data for plotting: {e}")
    sys.exit(1)

try:
    print("  Generating time series plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(t_H1, y_H1, label='H1 (Hanford)', color='C0', alpha=0.8)
    plt.plot(t_L1, y_L1, label='L1 (Livingston)', color='C1', alpha=0.8)
    plt.xlabel('Time (s) since GPS {}'.format(bandpassed_strain_H1.start_time))
    plt.ylabel('Strain (whitened, bandpassed)')
    plt.title('GW150914: Whitened & Bandpassed Strain Data (H1 & L1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("  Plot displayed successfully.")
except Exception as e:
    print(f"  Error generating plot: {e}")
    sys.exit(1)

print("TASK 4 complete.\n")
print("="*60)
print("All tasks completed successfully.")