# --- Imports ---
import os
import sys
import traceback
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# --- Constants and Configuration ---
EVENT_GPS_TIMES = {
    'GW150914': 1126259462.4,
    'GW151226': 1135136350.6,
    'GW170104': 1167559936.6,
    'GW170608': 1180922494.5,
    'GW170814': 1186741861.5,
    'GW190521': 1242442967.4,
}
DETECTORS = ['H1', 'L1']
DURATION = 64  # seconds (±32s)
HALF_DURATION = DURATION // 2
BANDPASS_LOW = 35
BANDPASS_HIGH = 350

# Output directories
PLOT_DIR = "plots"
SPECTROGRAM_DIR = "spectrograms"
DATA_DIR = "filtered_data"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- 1. Download Strain Data ---
print("="*60)
print("STEP 1: Downloading strain data from GWOSC")
print("="*60)
strain_data = {}

for event, gps_time in EVENT_GPS_TIMES.items():
    print(f"\nProcessing event: {event} (GPS: {gps_time})")
    strain_data[event] = {}
    start = gps_time - HALF_DURATION
    end = gps_time + HALF_DURATION
    for det in DETECTORS:
        print(f"  Fetching {det} data from {start} to {end} ...")
        try:
            ts = TimeSeries.fetch_open_data(det, start, end, cache=True)
            strain_data[event][det] = ts
            print(f"    Success: {det} data downloaded.")
        except Exception as e:
            print(f"    ERROR: Failed to fetch {det} data for {event}: {e}")
            traceback.print_exc()
            strain_data[event][det] = None

print("\nData loading complete. Strain data is stored in the 'strain_data' dictionary.")

# --- 2. Apply Bandpass Filter ---
print("\n" + "="*60)
print("STEP 2: Filtering strain data (35–350 Hz bandpass)")
print("="*60)
filtered_strain_data = {}

for event in strain_data:
    filtered_strain_data[event] = {}
    print(f"\nFiltering event: {event}")
    for det in strain_data[event]:
        ts = strain_data[event][det]
        if ts is None:
            print(f"  Skipping {det}: No data available.")
            filtered_strain_data[event][det] = None
            continue
        try:
            print(f"  Applying bandpass filter to {det} data ({BANDPASS_LOW}–{BANDPASS_HIGH} Hz)...")
            filtered_ts = ts.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
            filtered_strain_data[event][det] = filtered_ts
            # Save filtered data to disk for reproducibility
            fname = os.path.join(DATA_DIR, f"{event}_{det}_filtered.txt")
            filtered_ts.write(fname, format='ascii')
            print(f"    Success: {det} data filtered and saved to {fname}.")
        except Exception as e:
            print(f"    ERROR: Failed to filter {det} data for {event}: {e}")
            traceback.print_exc()
            filtered_strain_data[event][det] = None

print("\nFiltering complete. Filtered data is stored in the 'filtered_strain_data' dictionary.")

# --- 3. Time-Domain Plots ---
print("\n" + "="*60)
print("STEP 3: Creating time-domain plots (±0.2s around merger)")
print("="*60)
for event, gps_time in EVENT_GPS_TIMES.items():
    print(f"\nPlotting event: {event}")
    for det in DETECTORS:
        ts = filtered_strain_data.get(event, {}).get(det, None)
        if ts is None:
            print(f"  Skipping {det}: No filtered data available.")
            continue
        try:
            t0 = gps_time - 0.2
            t1 = gps_time + 0.2
            ts_window = ts.crop(t0, t1)
            times = ts_window.times.value - gps_time  # relative to merger

            plt.figure(figsize=(8, 4))
            plt.plot(times, ts_window.value, label=f"{event} {det}")
            plt.xlabel("Time (s) relative to merger")
            plt.ylabel("Strain")
            plt.title(f"{event} {det} | Filtered Strain ±0.2s around Merger")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(PLOT_DIR, f"{event}_{det}_strain.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  Success: Plotted {det} for {event}. Saved to {plot_path}")
        except Exception as e:
            print(f"  ERROR: Failed to plot {det} for {event}: {e}")
            traceback.print_exc()

# --- 4. Spectrograms ---
print("\n" + "="*60)
print("STEP 4: Generating time–frequency spectrograms (±2s around merger)")
print("="*60)
for event, gps_time in EVENT_GPS_TIMES.items():
    print(f"\nGenerating spectrogram for event: {event}")
    for det in DETECTORS:
        ts = filtered_strain_data.get(event, {}).get(det, None)
        if ts is None:
            print(f"  Skipping {det}: No filtered data available.")
            continue
        try:
            t0 = gps_time - 2
            t1 = gps_time + 2
            ts_window = ts.crop(t0, t1)
            print(f"  Computing spectrogram for {det}...")
            spec = ts_window.spectrogram(fftlength=0.125, overlap=0.0625)
            fig = spec.plot(norm='log', vmin=1e-24, vmax=1e-21)
            ax = fig.gca()
            ax.set_title(f"{event} {det} | Spectrogram ±2s around Merger")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_xlabel("Time [s] since GPS {:.1f}".format(gps_time))
            plt.tight_layout()
            spec_path = os.path.join(SPECTROGRAM_DIR, f"{event}_{det}_spectrogram.png")
            plt.savefig(spec_path)
            plt.close()
            print(f"  Success: Spectrogram plotted for {det} of {event}. Saved to {spec_path}")
        except Exception as e:
            print(f"  ERROR: Failed to generate spectrogram for {det} of {event}: {e}")
            traceback.print_exc()

print("\nAll tasks complete. Results saved in 'plots/', 'spectrograms/', and 'filtered_data/' directories.")