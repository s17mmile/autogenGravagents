# =========================
# GW170608 Strain Analysis
# =========================

# ---- Imports ----
import sys
import numpy as np
import matplotlib.pyplot as plt
from pycbc.frame import query_and_read_frame
from pycbc.types import TimeSeries
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

# ---- Event and Segment Info ----
event_time = 1180922494.5  # GW170608 GPS time
segment_duration = 4       # seconds
start_time = event_time - segment_duration / 2
end_time = event_time + segment_duration / 2

# ---- Task 1: Download Strain Data ----
print("="*60)
print("Task 1: Downloading strain data for GW170608 (H1 and L1)...")
detectors = ['H1', 'L1']
strain_data = {}

for det in detectors:
    print(f"  Downloading strain data for {det} from {start_time} to {end_time}...")
    try:
        # Query and read the frame from LOSC (no urltype argument)
        ts = query_and_read_frame(
            det, det,  # detector and channel are the same for LOSC
            start_time, end_time
        )
        strain_data[det] = ts
        print(f"    Successfully downloaded strain data for {det}.")
    except Exception as e:
        print(f"    Error downloading strain data for {det}: {e}")
        strain_data[det] = None

# Assign to variables for downstream tasks
strain_H1 = strain_data['H1']
strain_L1 = strain_data['L1']

# Optionally save strain data to disk for reproducibility
try:
    if strain_H1 is not None:
        strain_H1.save_to_wav('GW170608_H1_strain.wav')
        print("  H1 strain data saved to GW170608_H1_strain.wav")
    if strain_L1 is not None:
        strain_L1.save_to_wav('GW170608_L1_strain.wav')
        print("  L1 strain data saved to GW170608_L1_strain.wav")
except Exception as e:
    print(f"  Warning: Could not save strain data to WAV: {e}")

# ---- Task 2: Plot Raw Strain Time Series ----
print("="*60)
print("Task 2: Plotting raw strain time series for H1 and L1...")

if strain_H1 is None or strain_L1 is None:
    print("  Error: Strain data for one or both detectors is missing. Cannot plot.")
else:
    try:
        # Prepare time arrays relative to event time
        time_H1 = strain_H1.sample_times - event_time
        time_L1 = strain_L1.sample_times - event_time

        # Create figure and axes
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Plot H1
        axs[0].plot(time_H1, strain_H1, color='C0')
        axs[0].axvline(0, color='r', linestyle='--', label='GW170608 Event')
        axs[0].set_ylabel('Strain')
        axs[0].set_title('H1 Strain Data')
        axs[0].legend()
        
        # Plot L1
        axs[1].plot(time_L1, strain_L1, color='C1')
        axs[1].axvline(0, color='r', linestyle='--', label='GW170608 Event')
        axs[1].set_xlabel('Time (s) relative to event')
        axs[1].set_ylabel('Strain')
        axs[1].set_title('L1 Strain Data')
        axs[1].legend()
        
        plt.tight_layout()
        print("  Displaying strain time series plots interactively...")
        plt.show()

        # Save the plot after display
        fig.savefig("GW170608_strain_timeseries.png")
        print("  Strain time series plot saved as GW170608_strain_timeseries.png")
    except Exception as e:
        print(f"  Error while plotting strain data: {e}")

# ---- Task 3: Q-transform Spectrogram ----
print("="*60)
print("Task 3: Generating Q-transform spectrogram for H1...")

detector = 'H1'
strain = strain_H1

if strain is None:
    print(f"  Error: Strain data for {detector} is missing. Cannot compute Q-transform.")
else:
    try:
        print(f"  Converting PyCBC TimeSeries to GWpy TimeSeries for {detector}...")
        # Convert PyCBC TimeSeries to GWpy TimeSeries
        gwpy_strain = GWpyTimeSeries(strain.numpy(), sample_rate=strain.sample_rate, t0=strain.start_time)
        
        print("  Computing Q-transform (this may take a few seconds)...")
        # Compute Q-transform
        q = gwpy_strain.q_transform(outseg=(event_time-2, event_time+2))
        
        print("  Plotting Q-transform spectrogram interactively...")
        # Plot Q-transform
        fig = q.plot(figsize=(10, 6))
        ax = fig.gca()
        ax.axvline(event_time, color='r', linestyle='--', label='GW170608 Event')
        ax.set_title(f'{detector} Q-transform Spectrogram around GW170608')
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Save the Q-transform plot after display
        fig.savefig("GW170608_H1_qtransform.png")
        print("  Q-transform spectrogram saved as GW170608_H1_qtransform.png")
    except Exception as e:
        print(f"  Error during Q-transform computation or plotting: {e}")

print("="*60)
print("Workflow complete.")