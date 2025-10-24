# ==========================================================
# GW170608 Strain Data Analysis with GWpy (H1 and L1)
# ==========================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

# ---- Event and Analysis Parameters ----
event_time = 1180922494.5  # GW170608 GPS time
window = 32                # seconds before and after event for data fetch
start_time = event_time - window
end_time = event_time + window

# Bandpass filter parameters
low_freq = 35
high_freq = 350

# Time-domain plot window (±0.2s)
plot_window = 0.2
plot_start = event_time - plot_window
plot_end = event_time + plot_window

# Q-transform parameters
q_window = 0.5  # seconds before and after event
q_start = event_time - q_window
q_end = event_time + q_window
fmin = 20
fmax = 400

# ---- Task 1: Fetch Strain Data from GWOSC ----
print("="*60)
print("Task 1: Fetching ±32s strain data for GW170608 (H1 and L1) from GWOSC...")
gwpy_strain_H1 = None
gwpy_strain_L1 = None

for det in ['H1', 'L1']:
    try:
        print(f"  Fetching {det} strain data from {start_time} to {end_time} (GPS)...")
        strain = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        if det == 'H1':
            gwpy_strain_H1 = strain
        else:
            gwpy_strain_L1 = strain
        print(f"    Successfully fetched {det} strain data.")
        # Optionally save raw data
        strain.write(f"GW170608_{det}_raw.hdf5", format='hdf5')
        print(f"    Raw {det} strain data saved to GW170608_{det}_raw.hdf5")
    except Exception as e:
        print(f"    Error fetching {det} strain data: {e}")

# ---- Task 2: Bandpass Filter ----
print("="*60)
print(f"Task 2: Applying bandpass filter ({low_freq}-{high_freq} Hz) to strain data...")
filtered_strain_H1 = None
filtered_strain_L1 = None

# Filter H1 data
if gwpy_strain_H1 is not None:
    try:
        print("  Applying bandpass filter to H1 strain data (35-350 Hz)...")
        filtered_strain_H1 = gwpy_strain_H1.bandpass(low_freq, high_freq)
        print("    H1 strain data filtered successfully.")
        filtered_strain_H1.write("GW170608_H1_filtered.hdf5", format='hdf5')
        print("    Filtered H1 strain data saved to GW170608_H1_filtered.hdf5")
    except Exception as e:
        print(f"    Error filtering H1 strain data: {e}")
else:
    print("  Warning: gwpy_strain_H1 not available. Skipping H1 filtering.")

# Filter L1 data
if gwpy_strain_L1 is not None:
    try:
        print("  Applying bandpass filter to L1 strain data (35-350 Hz)...")
        filtered_strain_L1 = gwpy_strain_L1.bandpass(low_freq, high_freq)
        print("    L1 strain data filtered successfully.")
        filtered_strain_L1.write("GW170608_L1_filtered.hdf5", format='hdf5')
        print("    Filtered L1 strain data saved to GW170608_L1_filtered.hdf5")
    except Exception as e:
        print(f"    Error filtering L1 strain data: {e}")
else:
    print("  Warning: gwpy_strain_L1 not available. Skipping L1 filtering.")

# ---- Task 3: Time-Domain Plot ----
print("="*60)
print(f"Task 3: Plotting filtered strain data in time domain (±{plot_window}s around merger)...")

def plot_time_domain(strain, det, event_time, plot_start, plot_end):
    if strain is None:
        print(f"  Warning: No filtered strain data for {det}, skipping plot.")
        return
    try:
        print(f"  Plotting time-domain strain for {det} from {plot_start} to {plot_end} s...")
        cropped = strain.crop(plot_start, plot_end)
        plt.figure(figsize=(10, 4))
        plt.plot(cropped.times.value, cropped.value, label=f'{det} strain')
        plt.axvline(event_time, color='r', linestyle='--', label='Merger Time')
        plt.xlabel('GPS Time [s]')
        plt.ylabel('Strain')
        plt.title(f'{det} Strain around GW170608 Merger')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f"GW170608_{det}_filtered_time_domain.png")
        print(f"  Time-domain plot for {det} saved as GW170608_{det}_filtered_time_domain.png")
    except Exception as e:
        print(f"  Error plotting time-domain strain for {det}: {e}")

plot_time_domain(filtered_strain_H1, 'H1', event_time, plot_start, plot_end)
plot_time_domain(filtered_strain_L1, 'L1', event_time, plot_start, plot_end)

# ---- Task 4: Q-transform Spectrogram ----
print("="*60)
print(f"Task 4: Generating Q-transform spectrograms (±{q_window}s, {fmin}-{fmax} Hz)...")

def plot_qtransform(strain, det, event_time, q_start, q_end, fmin, fmax):
    if strain is None:
        print(f"  Warning: No filtered strain data for {det}, skipping Q-transform.")
        return
    try:
        print(f"  Computing Q-transform for {det} from {q_start} to {q_end} s...")
        cropped = strain.crop(q_start, q_end)
        q = cropped.q_transform(outseg=(q_start, q_end), frange=(fmin, fmax))
        print(f"  Plotting Q-transform spectrogram for {det}...")
        fig = q.plot()
        ax = fig.gca()
        ax.axvline(event_time, color='r', linestyle='--', label='Merger Time')
        ax.set_title(f'{det} Q-transform Spectrogram around GW170608')
        ax.set_xlabel('GPS Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        ax.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(f"GW170608_{det}_qtransform.png")
        print(f"  Q-transform spectrogram for {det} saved as GW170608_{det}_qtransform.png")
    except Exception as e:
        print(f"  Error generating Q-transform for {det}: {e}")

plot_qtransform(filtered_strain_H1, 'H1', event_time, q_start, q_end, fmin, fmax)
plot_qtransform(filtered_strain_L1, 'L1', event_time, q_start, q_end, fmin, fmax)

print("="*60)
print("Workflow complete.")