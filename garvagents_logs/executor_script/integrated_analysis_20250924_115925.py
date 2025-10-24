# =========================
# Gravitational Wave Event Analysis Pipeline
# =========================

# ---- Imports ----
import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

# ---- Configuration ----
# List of events and their GPS times
EVENTS = {
    "GW150914": 1126259462,
    "GW151226": 1135136350,
    "GW170104": 1167559936,
    "GW170608": 1180922494,
    "GW170814": 1186741861,
    "GW190521": 1242442967,
}
DETECTORS = ['H1', 'L1']
DATA_WINDOW = 2048  # seconds for data download
FILTER_LOW = 35     # Hz
FILTER_HIGH = 350   # Hz
TIME_PLOT_WINDOW = 0.2  # seconds for time-domain plot
SPEC_WINDOW = 1.0       # seconds for spectrogram

# Output directories
PLOT_DIR = "plots"
SPEC_DIR = "spectrograms"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SPEC_DIR, exist_ok=True)

# =========================
# 1. Data Loading
# =========================
print("\n=== [1/4] Downloading strain data from GWOSC ===")
strain_data = {}

for event, gps_time in EVENTS.items():
    print(f"\nProcessing event: {event} (GPS: {gps_time})")
    strain_data[event] = {}
    for det in DETECTORS:
        start = gps_time - DATA_WINDOW
        end = gps_time + DATA_WINDOW
        print(f"  Fetching data for {det} from {start} to {end}...")
        try:
            ts = TimeSeries.fetch_open_data(det, start, end, verbose=True)
            strain_data[event][det] = ts
            print(f"    Success: {det} data loaded for {event}.")
        except Exception as e:
            print(f"    ERROR: Could not fetch {det} data for {event}: {e}")
            strain_data[event][det] = None

print("\nData loading complete.")

# =========================
# 2. Bandpass Filtering
# =========================
print("\n=== [2/4] Applying bandpass filter ({}–{} Hz) ===".format(FILTER_LOW, FILTER_HIGH))
filtered_strain_data = {}

for event, detectors in strain_data.items():
    print(f"\nFiltering event: {event}")
    filtered_strain_data[event] = {}
    for det, ts in detectors.items():
        if ts is None:
            print(f"  Skipping {det} for {event}: No data available.")
            filtered_strain_data[event][det] = None
            continue
        try:
            print(f"  Applying bandpass filter to {det} data...")
            filtered_ts = ts.bandpass(FILTER_LOW, FILTER_HIGH)
            filtered_strain_data[event][det] = filtered_ts
            print(f"    Success: {det} data filtered for {event}.")
            # Optional: Save filtered data as numpy array for persistence
            np.save(os.path.join(PLOT_DIR, f"{event}_{det}_filtered.npy"), filtered_ts.value)
        except Exception as e:
            print(f"    ERROR: Could not filter {det} data for {event}: {e}")
            filtered_strain_data[event][det] = None

print("\nBandpass filtering complete.")

# =========================
# 3. Time-Domain Plotting
# =========================
print("\n=== [3/4] Creating time-domain plots (±{}s around merger) ===".format(TIME_PLOT_WINDOW))

for event, detectors in filtered_strain_data.items():
    gps_time = EVENTS[event]
    print(f"\nPlotting event: {event} (GPS: {gps_time})")
    for det, ts in detectors.items():
        if ts is None:
            print(f"  Skipping {det} for {event}: No filtered data available.")
            continue
        try:
            # Crop to ±0.2 seconds around merger
            cropped_ts = ts.crop(gps_time - TIME_PLOT_WINDOW, gps_time + TIME_PLOT_WINDOW)
            times = cropped_ts.times.value - gps_time  # Relative to merger
            plt.figure(figsize=(10, 4))
            plt.plot(times, cropped_ts.value, label=f'{det}')
            plt.title(f'{event} {det} Strain (±{TIME_PLOT_WINDOW}s around merger)')
            plt.xlabel('Time (s) relative to merger')
            plt.ylabel('Strain')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            # Save plot
            plot_path = os.path.join(PLOT_DIR, f"{event}_{det}_timedomain.png")
            plt.savefig(plot_path)
            plt.show()
            print(f"    Success: Plotted {det} for {event}. Saved to {plot_path}")
        except Exception as e:
            print(f"    ERROR: Could not plot {det} for {event}: {e}")

print("\nTime-domain plotting complete.")

# =========================
# 4. Spectrogram Generation
# =========================
print("\n=== [4/4] Generating time–frequency spectrograms (±{}s around merger) ===".format(SPEC_WINDOW))

for event, detectors in filtered_strain_data.items():
    gps_time = EVENTS[event]
    print(f"\nGenerating spectrograms for event: {event} (GPS: {gps_time})")
    for det, ts in detectors.items():
        if ts is None:
            print(f"  Skipping {det} for {event}: No filtered data available.")
            continue
        try:
            # Crop to ±1 second around merger
            cropped_ts = ts.crop(gps_time - SPEC_WINDOW, gps_time + SPEC_WINDOW)
            print(f"  Computing spectrogram for {det}...")
            # Use 0.125s segments for better time resolution
            spec = cropped_ts.spectrogram(fftlength=0.125)
            # Convert time axis to seconds relative to merger
            rel_times = spec.times.value - gps_time
            plt.figure(figsize=(10, 5))
            plt.pcolormesh(rel_times, spec.frequencies.value, spec.value.T, shading='auto', cmap='viridis')
            plt.title(f'{event} {det} Spectrogram (±{SPEC_WINDOW}s around merger)')
            plt.xlabel('Time (s) relative to merger')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='ASD [strain/√Hz]')
            plt.ylim(20, 400)  # Focus on GW band
            plt.tight_layout()
            # Save spectrogram
            spec_path = os.path.join(SPEC_DIR, f"{event}_{det}_spectrogram.png")
            plt.savefig(spec_path)
            plt.show()
            print(f"    Success: Spectrogram plotted for {det} of {event}. Saved to {spec_path}")
        except Exception as e:
            print(f"    ERROR: Could not generate spectrogram for {det} of {event}: {e}")

print("\nSpectrogram generation complete.")

print("\n=== All tasks complete. Results saved in '{}' and '{}' directories. ===".format(PLOT_DIR, SPEC_DIR))