# --- Imports ---
import sys
import numpy as np
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.frame import query_and_read_frame
from pycbc.types import TimeSeries
from pycbc.psd import welch

# --- Section 1: Download GW150914 H1 and L1 Strain Data ---
print("="*60)
print("Step 1: Downloading GW150914 H1 and L1 strain data...")
event_name = "GW150914"
padding = 5  # seconds

try:
    merger = Merger(event_name)
    event_time = merger.time
    print(f"Event GPS time: {event_time}")

    # Define start and end times with padding
    start = event_time - padding
    end = event_time + padding
    print(f"Data segment: {start} to {end} (GPS seconds)")

    detectors = ['H1', 'L1']
    strain_data = {}

    for det in detectors:
        print(f"  Downloading strain data for {det}...")
        try:
            # Correct usage: pass channel name as first positional argument
            channel_name = f"{det}:LOSC-STRAIN"
            ts = query_and_read_frame(
                channel_name,
                start,
                end
            )
            strain_data[det] = ts
            print(f"    Successfully downloaded {det} strain data.")
        except Exception as e:
            print(f"    Error downloading strain data for {det}: {e}")
            strain_data[det] = None

    strain_H1 = strain_data['H1']
    strain_L1 = strain_data['L1']

    if strain_H1 is None or strain_L1 is None:
        raise RuntimeError("Failed to download strain data for one or both detectors.")

    # Optionally save raw data
    # strain_H1.save('GW150914_H1_raw.npy')
    # strain_L1.save('GW150914_L1_raw.npy')

except Exception as main_e:
    print(f"Failed to download GW150914 strain data: {main_e}")
    sys.exit(1)

print("Step 1 complete.")
print("="*60)

# --- Section 2: Whitening the Strain Data ---
print("Step 2: Whitening the strain data...")
try:
    assert strain_H1 is not None and strain_L1 is not None
except AssertionError:
    print("Strain data not loaded. Exiting.")
    sys.exit(1)

psd_segment_length = 4  # seconds
psd_avg_method = 'median'
psd_fftlength = psd_segment_length
psd_overlap = psd_fftlength // 2

whitened_data = {}

for det, strain in [('H1', strain_H1), ('L1', strain_L1)]:
    try:
        print(f"  Estimating PSD for {det}...")
        psd = welch(
            strain,
            seg_len=psd_fftlength,
            avg_method=psd_avg_method,
            overlap=psd_overlap
        )
        print(f"    PSD estimated for {det}.")

        print(f"  Whitening strain data for {det}...")
        whitened = strain.whiten(4, 2, psd=psd)
        whitened_data[det] = whitened
        print(f"    Strain data whitened for {det}.")

        # Optionally save whitened data
        # np.save(f'GW150914_{det}_whitened.npy', whitened.numpy())

    except Exception as e:
        print(f"    Error processing {det}: {e}")
        whitened_data[det] = None

whitened_H1 = whitened_data['H1']
whitened_L1 = whitened_data['L1']

if whitened_H1 is None or whitened_L1 is None:
    print("Whitening failed for one or both detectors. Exiting.")
    sys.exit(1)

print("Step 2 complete.")
print("="*60)

# --- Section 3: Bandpass Filtering (30–250 Hz) ---
print("Step 3: Applying bandpass filter (30–250 Hz)...")
try:
    assert whitened_H1 is not None and whitened_L1 is not None
except AssertionError:
    print("Whitened data not available. Exiting.")
    sys.exit(1)

low_freq = 30.0   # Hz
high_freq = 250.0 # Hz

filtered_data = {}

for det, whitened in [('H1', whitened_H1), ('L1', whitened_L1)]:
    try:
        print(f"  Applying highpass filter at {low_freq} Hz to {det}...")
        hp = whitened.highpass(low_freq)
        print(f"    Highpass filter applied to {det}.")

        print(f"  Applying lowpass filter at {high_freq} Hz to {det}...")
        bp = hp.lowpass(high_freq)
        print(f"    Lowpass filter applied to {det}.")

        filtered_data[det] = bp
        print(f"    Bandpass filtering complete for {det}.")

        # Optionally save filtered data
        # np.save(f'GW150914_{det}_filtered.npy', bp.numpy())

    except Exception as e:
        print(f"    Error filtering {det}: {e}")
        filtered_data[det] = None

filtered_H1 = filtered_data['H1']
filtered_L1 = filtered_data['L1']

if filtered_H1 is None or filtered_L1 is None:
    print("Filtering failed for one or both detectors. Exiting.")
    sys.exit(1)

print("Step 3 complete.")
print("="*60)

# --- Section 4: Plotting the Processed Strain Data ---
print("Step 4: Plotting processed H1 and L1 strain data...")
try:
    # Prepare time axes relative to event time
    t_H1 = filtered_H1.sample_times - event_time
    t_L1 = filtered_L1.sample_times - event_time

    plt.figure(figsize=(10, 6))
    plt.plot(t_H1, filtered_H1, label='H1', color='C0', alpha=0.8)
    plt.plot(t_L1, filtered_L1, label='L1', color='C1', alpha=0.8)
    plt.axvline(0, color='k', linestyle='--', label='GW150914 Event Time')
    plt.xlabel('Time (s) relative to GW150914')
    plt.ylabel('Whitened, Bandpassed Strain')
    plt.title('GW150914: H1 and L1 Processed Strain Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("Plotting complete.")

    # Optionally save the plot
    # plt.savefig('GW150914_H1_L1_processed_strain.png', dpi=150)

except Exception as e:
    print(f"Error during plotting: {e}")
    sys.exit(1)

print("="*60)
print("Workflow complete. All steps executed successfully.")

# Remove any stray or redundant calls to query_and_read_frame here.
# All data acquisition is handled in the main workflow above.