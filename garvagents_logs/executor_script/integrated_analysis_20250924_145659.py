# ==========================================================
# GW170608 Strain Data Analysis with GWpy (H1 and L1)
# ==========================================================

# ---- Imports ----
import sys
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

# ---- Event and Analysis Parameters ----
event_time = 1180922494.5  # GW170608 GPS time
half_window = 32           # seconds for data fetch (±32s)
start_time = event_time - half_window
end_time = event_time + half_window

# Bandpass filter parameters
low_freq = 35
high_freq = 350

# Time-domain plot window (±0.2s)
plot_window = 0.2
plot_start = event_time - plot_window
plot_end = event_time + plot_window

# Q-transform parameters
q_window = 0.5  # seconds around the event
q_start = event_time - q_window
q_end = event_time + q_window
q_freq_min = 30
q_freq_max = 400
q_transform_kwargs = {
    'outseg': (q_start, q_end),
    'frange': (q_freq_min, q_freq_max),
    'qrange': (8, 64),
    'logf': True,
    'dt': 0.01,   # <-- Changed from 'stride' to 'dt'
    'pad': True,
}

# ---- Task 1: Fetch Strain Data from GWOSC ----
print("="*60)
print("Task 1: Fetching ±32s strain data for GW170608 (H1 and L1) from GWOSC...")
detectors = ['H1', 'L1']
gwpy_strain = {}

for det in detectors:
    print(f"  Fetching {det} strain data from {start_time} to {end_time} (GPS)...")
    try:
        strain = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        gwpy_strain[det] = strain
        print(f"    Successfully fetched {det} strain data.")
        # Optionally save raw data
        strain.write(f"GW170608_{det}_raw.hdf5", format='hdf5')
        print(f"    Raw {det} strain data saved to GW170608_{det}_raw.hdf5")
    except Exception as e:
        print(f"    Error fetching {det} strain data: {e}")
        gwpy_strain[det] = None

gwpy_strain_H1 = gwpy_strain['H1']
gwpy_strain_L1 = gwpy_strain['L1']

# ---- Task 2: Bandpass Filter ----
print("="*60)
print(f"Task 2: Applying bandpass filter ({low_freq}-{high_freq} Hz) to strain data...")
filtered_strain = {}

for det, strain in [('H1', gwpy_strain_H1), ('L1', gwpy_strain_L1)]:
    if strain is None:
        print(f"  Warning: No strain data for {det}, skipping bandpass filter.")
        filtered_strain[det] = None
        continue
    try:
        print(f"  Applying bandpass filter to {det}...")
        filtered = strain.bandpass(low_freq, high_freq)
        filtered_strain[det] = filtered
        print(f"    Bandpass filter applied successfully to {det}.")
        # Optionally save filtered data
        filtered.write(f"GW170608_{det}_filtered.hdf5", format='hdf5')
        print(f"    Filtered {det} strain data saved to GW170608_{det}_filtered.hdf5")
    except Exception as e:
        print(f"  Error applying bandpass filter to {det}: {e}")
        filtered_strain[det] = None

filtered_strain_H1 = filtered_strain['H1']
filtered_strain_L1 = filtered_strain['L1']

# ---- Task 3: Time-Domain Plot ----
print("="*60)
print(f"Task 3: Plotting filtered strain data in time domain (±{plot_window}s around merger)...")

if filtered_strain_H1 is None and filtered_strain_L1 is None:
    print("  Error: No filtered strain data available for either detector. Cannot plot.")
else:
    try:
        print("  Preparing time-domain plots for filtered strain data...")
        plt.figure(figsize=(12, 6))

        # Plot H1
        if filtered_strain_H1 is not None:
            print("    Plotting H1...")
            h1_segment = filtered_strain_H1.crop(plot_start, plot_end)
            plt.plot(
                h1_segment.times.value,
                h1_segment.value,
                label='H1',
                color='C0'
            )
        else:
            print("    Warning: No filtered H1 data to plot.")

        # Plot L1
        if filtered_strain_L1 is not None:
            print("    Plotting L1...")
            l1_segment = filtered_strain_L1.crop(plot_start, plot_end)
            plt.plot(
                l1_segment.times.value,
                l1_segment.value,
                label='L1',
                color='C1'
            )
        else:
            print("    Warning: No filtered L1 data to plot.")

        # Mark the merger time
        plt.axvline(event_time, color='k', linestyle='--', label='Merger Time')

        # Label and format
        plt.xlabel('GPS Time (s)')
        plt.ylabel('Strain')
        plt.title(f'Filtered Strain Data around GW170608 Merger (±{plot_window} s)')
        plt.legend()
        plt.tight_layout()
        print("  Displaying time-domain plot interactively...")
        plt.show()
        plt.savefig("GW170608_filtered_strain_time_domain.png")
        print("  Time-domain plot saved as GW170608_filtered_strain_time_domain.png")
    except Exception as e:
        print(f"  Error during time-domain plotting: {e}")

# ---- Task 4: Q-transform Spectrogram ----
print("="*60)
print(f"Task 4: Generating Q-transform spectrograms (±{q_window}s, {q_freq_min}-{q_freq_max} Hz)...")

for det, strain in [('H1', filtered_strain_H1), ('L1', filtered_strain_L1)]:
    if strain is None:
        print(f"  Warning: No filtered strain data for {det}, skipping Q-transform.")
        continue
    try:
        print(f"  Computing Q-transform for {det} in {q_start}–{q_end} s, {q_freq_min}–{q_freq_max} Hz...")
        segment = strain.crop(q_start, q_end)
        qspec = segment.q_transform(**q_transform_kwargs)
        print(f"    Q-transform computed for {det}. Plotting...")
        fig = qspec.plot(figsize=(10, 6), vmin=0.0001, vmax=0.01, cmap='viridis')
        ax = fig.gca()
        ax.axvline(event_time, color='r', linestyle='--', label='Merger Time')
        ax.set_title(f'{det} Q-transform Spectrogram around GW170608')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('GPS Time [s]')
        ax.legend()
        plt.tight_layout()
        print(f"  Displaying Q-transform spectrogram for {det} interactively...")
        plt.show()
        fig.savefig(f"GW170608_{det}_qtransform.png")
        print(f"  Q-transform spectrogram saved as GW170608_{det}_qtransform.png")
    except Exception as e:
        print(f"  Error computing or plotting Q-transform for {det}: {e}")

print("="*60)
print("Workflow complete.")