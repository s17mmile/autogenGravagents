# ================================
# GW170608 Strain Data Analysis
# ================================

# ----------- Imports ------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from gwpy.timeseries import TimeSeries as GWpy_TimeSeries

# ----------- Parameters ----------
EVENT_NAME = "GW170608"
DETECTOR = "L1"  # Livingston
WINDOW = 4  # seconds
RAW_PLOT_FILENAME = "GW170608_raw_strain_L1.png"
QTRANSFORM_PLOT_FILENAME = "GW170608_qtransform_L1.png"

# ----------- Task 1: Download Strain Data -----------
print("="*60)
print("TASK 1: Downloading strain data for GW170608...")
strain_data = None
event_time = None

try:
    print(f"Fetching event information for {EVENT_NAME}...")
    event = Merger(EVENT_NAME)
    event_time = event.time
    print(f"Event GPS time: {event_time}")

    # Define start and end times for the strain data
    start = event_time - WINDOW / 2
    end = event_time + WINDOW / 2

    print(f"Downloading strain data for {DETECTOR} from {start} to {end} (GPS)...")
    strain = event.strain(DETECTOR, start, end)
    print(f"Strain data downloaded: {strain}")

    strain_data = strain  # PyCBC TimeSeries

except Exception as e:
    print(f"Error downloading strain data: {e}")
    sys.exit(1)

# ----------- Task 2: Plot Raw Strain Time Series -----------
print("="*60)
print("TASK 2: Plotting raw strain time series...")

try:
    if strain_data is None:
        raise ValueError("No strain data available. Please run the data loading step first.")

    # Extract time and strain arrays
    times = strain_data.sample_times
    strain_array = strain_data.numpy() if hasattr(strain_data, 'numpy') else np.array(strain_data)

    # Event time for marking (center of window)
    event_time_plot = strain_data.start_time + (len(strain_data) / strain_data.sample_rate) / 2

    plt.figure(figsize=(10, 5))
    plt.plot(times, strain_array, label='Strain')
    plt.axvline(x=event_time_plot, color='r', linestyle='--', label='GW170608 Event Time')
    plt.xlabel('Time (GPS seconds)')
    plt.ylabel('Strain')
    plt.title('GW170608 Raw Strain Time Series (Livingston)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Raw strain time series plot displayed successfully.")

    # Save the plot
    plt.figure(figsize=(10, 5))
    plt.plot(times, strain_array, label='Strain')
    plt.axvline(x=event_time_plot, color='r', linestyle='--', label='GW170608 Event Time')
    plt.xlabel('Time (GPS seconds)')
    plt.ylabel('Strain')
    plt.title('GW170608 Raw Strain Time Series (Livingston)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RAW_PLOT_FILENAME)
    plt.close()
    print(f"Raw strain time series plot saved as '{RAW_PLOT_FILENAME}'.")

except Exception as e:
    print(f"Error plotting strain time series: {e}")
    sys.exit(1)

# ----------- Task 3: Q-transform Spectrogram -----------
print("="*60)
print("TASK 3: Generating Q-transform spectrogram...")

try:
    if strain_data is None:
        raise ValueError("No strain data available. Please run the data loading step first.")

    print("Converting PyCBC TimeSeries to GWpy TimeSeries (if necessary)...")
    # Convert PyCBC TimeSeries to GWpy TimeSeries if needed
    if not isinstance(strain_data, GWpy_TimeSeries):
        gwpy_strain = GWpy_TimeSeries(strain_data.numpy(),
                                      times=strain_data.sample_times,
                                      name=f'{DETECTOR}:GWOSC-4KHZ_R1_STRAIN')
    else:
        gwpy_strain = strain_data

    print("Performing Q-transform...")
    q = gwpy_strain.q_transform(outseg=(gwpy_strain.t0.value, gwpy_strain.t1.value),
                               qrange=(8, 64),
                               frange=(20, 512),
                               logf=True)

    print("Plotting Q-transform spectrogram...")
    fig = q.plot(figsize=(10, 6), vmin=0, vmax=15)
    ax = fig.gca()
    # Mark the event time (center of window)
    event_time_q = gwpy_strain.t0.value + (gwpy_strain.t1.value - gwpy_strain.t0.value) / 2
    ax.axvline(event_time_q, color='r', linestyle='--', label='GW170608 Event Time')
    ax.set_title('GW170608 Q-transform Spectrogram (Livingston)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    print("Q-transform spectrogram displayed successfully.")

    # Save the Q-transform plot
    fig = q.plot(figsize=(10, 6), vmin=0, vmax=15)
    ax = fig.gca()
    ax.axvline(event_time_q, color='r', linestyle='--', label='GW170608 Event Time')
    ax.set_title('GW170608 Q-transform Spectrogram (Livingston)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(QTRANSFORM_PLOT_FILENAME)
    plt.close(fig)
    print(f"Q-transform spectrogram saved as '{QTRANSFORM_PLOT_FILENAME}'.")

except Exception as e:
    print(f"Error generating Q-transform spectrogram: {e}")
    sys.exit(1)

print("="*60)
print("All tasks completed successfully.")