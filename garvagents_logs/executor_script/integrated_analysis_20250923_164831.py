# ============================================================
# GW150914 H1 & L1 Strain Data: Download, Whiten, Filter, Plot
# ============================================================

# -------------------------
# Imports and Configuration
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.types import TimeSeries

# -------------------------
# 1. Data Download Section
# -------------------------
print("\n=== [1/4] Downloading GW150914 H1 and L1 strain data ===")
strain_H1 = None
strain_L1 = None
event = None
try:
    # Fetch event info
    print("Fetching GW150914 event information from PyCBC catalog...")
    event = Merger("GW150914")
    print(f"Event time (GPS): {event.time}")

    # Define segment: 32 seconds centered on event
    duration = 32  # seconds
    start = event.time - duration / 2
    end = event.time + duration / 2

    print("Downloading H1 strain data...")
    strain_H1 = event.strain('H1', start, end)
    print("H1 strain data downloaded successfully.")

    print("Downloading L1 strain data...")
    strain_L1 = event.strain('L1', start, end)
    print("L1 strain data downloaded successfully.")

    # Save raw data for reproducibility
    np.save("GW150914_H1_raw.npy", strain_H1.numpy())
    np.save("GW150914_L1_raw.npy", strain_L1.numpy())
except Exception as e:
    print(f"[ERROR] Failed to download strain data: {e}")
    exit(1)

# -------------------------
# 2. Whitening Section
# -------------------------
print("\n=== [2/4] Whitening strain data ===")
whitened_H1 = None
whitened_L1 = None
try:
    # Parameters for PSD estimation and whitening
    psd_duration = 16  # seconds for PSD estimation
    psd_stride = 8     # seconds to step away from the event to avoid signal
    low_frequency_cutoff = 20.0  # Hz

    # H1 whitening
    print("Estimating PSD for H1...")
    psd_start_H1 = strain_H1.start_time - psd_stride - psd_duration
    psd_end_H1 = psd_start_H1 + psd_duration
    psd_segment_H1 = strain_H1.time_slice(psd_start_H1, psd_end_H1)
    psd_H1 = psd_segment_H1.psd(psd_duration, avg_method='median')
    print("Whitening H1 strain data...")
    whitened_H1 = strain_H1.whiten(4, 4, psd=psd_H1, low_frequency_cutoff=low_frequency_cutoff)
    print("H1 strain data whitened.")

    # L1 whitening
    print("Estimating PSD for L1...")
    psd_start_L1 = strain_L1.start_time - psd_stride - psd_duration
    psd_end_L1 = psd_start_L1 + psd_duration
    psd_segment_L1 = strain_L1.time_slice(psd_start_L1, psd_end_L1)
    psd_L1 = psd_segment_L1.psd(psd_duration, avg_method='median')
    print("Whitening L1 strain data...")
    whitened_L1 = strain_L1.whiten(4, 4, psd=psd_L1, low_frequency_cutoff=low_frequency_cutoff)
    print("L1 strain data whitened.")

    # Save whitened data
    np.save("GW150914_H1_whitened.npy", whitened_H1.numpy())
    np.save("GW150914_L1_whitened.npy", whitened_L1.numpy())
except Exception as e:
    print(f"[ERROR] Whitening process failed: {e}")
    exit(1)

# -------------------------
# 3. Filtering Section
# -------------------------
print("\n=== [3/4] Bandpass filtering (30–250 Hz) ===")
filtered_H1 = None
filtered_L1 = None
try:
    # H1 filtering
    print("Applying highpass filter at 30 Hz to H1...")
    hp_H1 = whitened_H1.highpass(30.0)
    print("Applying lowpass filter at 250 Hz to H1...")
    filtered_H1 = hp_H1.lowpass(250.0)
    print("H1 data bandpassed between 30 Hz and 250 Hz.")

    # L1 filtering
    print("Applying highpass filter at 30 Hz to L1...")
    hp_L1 = whitened_L1.highpass(30.0)
    print("Applying lowpass filter at 250 Hz to L1...")
    filtered_L1 = hp_L1.lowpass(250.0)
    print("L1 data bandpassed between 30 Hz and 250 Hz.")

    # Save filtered data
    np.save("GW150914_H1_filtered.npy", filtered_H1.numpy())
    np.save("GW150914_L1_filtered.npy", filtered_L1.numpy())
except Exception as e:
    print(f"[ERROR] Filtering process failed: {e}")
    exit(1)

# -------------------------
# 4. Visualization Section
# -------------------------
print("\n=== [4/4] Plotting processed strain data ===")
try:
    print("Preparing data for plotting...")
    # Extract time and strain arrays
    times_H1 = filtered_H1.sample_times
    strain_H1_plot = filtered_H1.numpy()
    times_L1 = filtered_L1.sample_times
    strain_L1_plot = filtered_L1.numpy()

    print("Creating overlay plot of H1 and L1 processed strain data...")
    plt.figure(figsize=(10, 6))
    plt.plot(times_H1, strain_H1_plot, label='H1', color='C0', alpha=0.7)
    plt.plot(times_L1, strain_L1_plot, label='L1', color='C1', alpha=0.7)
    plt.xlabel('Time (s, GPS)')
    plt.ylabel('Whitened, Bandpassed Strain')
    plt.title('GW150914: Whitened and Bandpassed Strain Data (H1 & L1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GW150914_H1_L1_whitened_bandpassed.png", dpi=150)
    plt.show()
    print("Plot displayed and saved as 'GW150914_H1_L1_whitened_bandpassed.png'.")
except Exception as e:
    print(f"[ERROR] Visualization failed: {e}")

print("\n=== Workflow complete! ===")