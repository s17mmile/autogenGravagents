# --- GW170608 Gravitational Wave Data Analysis Integrated Script ---

import sys
import numpy as np
import matplotlib.pyplot as plt
from pycbc.types import TimeSeries
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

# ----------------------------- PARAMETERS -----------------------------
event_gps = 1180922494.5
window = 32  # seconds before and after event for download
zoom_window = 0.5  # seconds before and after event for zoomed plots
low_freq = 30.0
high_freq = 300.0
q_freq_min = 20
q_freq_max = 400
detectors = ['H1', 'L1']

output_files = {
    'H1': 'GW170608_H1_strain_zoom.png',
    'L1': 'GW170608_L1_strain_zoom.png'
}
q_output_files = {
    'H1': 'GW170608_H1_qtransform_zoom.png',
    'L1': 'GW170608_L1_qtransform_zoom.png'
}

# ----------------------------- TASK 1: DATA LOADING -----------------------------
print("="*60)
print("TASK 1: Downloading strain data for GW170608 (H1, L1)...")
start_time = event_gps - window
end_time = event_gps + window
strain_data = {}

for det in detectors:
    print(f"Downloading strain data for {det} from {start_time} to {end_time} (GPS)...")
    try:
        strain = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        strain_data[det] = strain
        print(f"  Successfully downloaded strain data for {det}.")
    except Exception as e:
        print(f"  Error downloading data for {det}: {e}", file=sys.stderr)
        strain_data[det] = None

# ----------------------------- TASK 2: PROCESSING -----------------------------
print("="*60)
print("TASK 2: Processing (bandpass filter and whitening)...")
processed_strain = {}

for det in detectors:
    print(f"Processing strain data for {det}...")
    raw_strain = strain_data.get(det)
    if raw_strain is None:
        print(f"  No data available for {det}, skipping.")
        processed_strain[det] = None
        continue
    try:
        print(f"  Applying bandpass filter ({low_freq}-{high_freq} Hz)...")
        filtered = raw_strain.highpass(low_freq)
        filtered = filtered.lowpass(high_freq)
        print("  Whitening data...")
        whitened = filtered.whiten(4, 2)
        processed_strain[det] = whitened
        print(f"  Processing complete for {det}.")
    except Exception as e:
        print(f"  Error processing data for {det}: {e}")
        processed_strain[det] = None

# ----------------------------- TASK 3: VISUALIZATION (STRAIN) -----------------------------
print("="*60)
print("TASK 3: Plotting filtered & whitened strain (zoomed ±0.5s)...")

for det in detectors:
    print(f"Preparing plot for {det}...")
    strain = processed_strain.get(det)
    if strain is None:
        print(f"  No processed data for {det}, skipping plot.")
        continue
    try:
        print(f"  Cropping to ±{zoom_window}s around GPS {event_gps}...")
        zoom_strain = strain.crop(event_gps - zoom_window, event_gps + zoom_window)
        times = zoom_strain.sample_times - event_gps  # Relative to event time

        plt.figure(figsize=(10, 4))
        plt.plot(times, zoom_strain, label=f'{det} strain', color='C0' if det == 'H1' else 'C1')
        plt.title(f'{det} Filtered & Whitened Strain around GW170608')
        plt.xlabel('Time (s) relative to event')
        plt.ylabel('Whitened strain')
        plt.axvline(0, color='k', linestyle='--', alpha=0.5, label='Event time')
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_files[det])
        print(f"  Plot saved to {output_files[det]}.")
        plt.show()
        plt.close()
    except Exception as e:
        print(f"  Error plotting for {det}: {e}")

# ----------------------------- TASK 4: VISUALIZATION (Q-TRANSFORM) -----------------------------
print("="*60)
print("TASK 4: Q-transform spectrogram (zoomed ±0.5s, 20–400 Hz)...")

for det in detectors:
    print(f"Preparing Q-transform spectrogram for {det}...")
    strain = processed_strain.get(det)
    if strain is None:
        print(f"  No processed data for {det}, skipping Q-transform.")
        continue
    try:
        print(f"  Cropping to ±{zoom_window}s around GPS {event_gps}...")
        zoom_strain = strain.crop(event_gps - zoom_window, event_gps + zoom_window)
        print("  Converting to GWpy TimeSeries...")
        gwpy_strain = GWpyTimeSeries(zoom_strain.numpy(),
                                     sample_rate=zoom_strain.sample_rate,
                                     t0=zoom_strain.start_time)
        print(f"  Computing Q-transform ({q_freq_min}-{q_freq_max} Hz)...")
        qspec = gwpy_strain.q_transform(frange=(q_freq_min, q_freq_max))
        print("  Plotting Q-transform spectrogram...")
        fig = qspec.plot(figsize=(10, 4))
        ax = fig.gca()
        ax.axvline(event_gps, color='w', linestyle='--', alpha=0.7, label='Event time')
        ax.set_xlim(event_gps - zoom_window, event_gps + zoom_window)
        ax.set_ylim(q_freq_min, q_freq_max)
        ax.set_title(f'{det} Q-transform Spectrogram around GW170608')
        ax.set_xlabel('Time (GPS)')
        ax.set_ylabel('Frequency [Hz]')
        ax.legend()
        plt.tight_layout()
        fig.savefig(q_output_files[det])
        print(f"  Q-transform spectrogram saved to {q_output_files[det]}.")
        plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"  Error generating Q-transform for {det}: {e}")

print("="*60)
print("Analysis complete. All plots have been saved and displayed.")