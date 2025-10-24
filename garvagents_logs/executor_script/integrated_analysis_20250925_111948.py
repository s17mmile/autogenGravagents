# --- Imports ---
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# --- Constants ---
EVENT_TIME = 1180922494.5
START_TIME = int(EVENT_TIME - 32)
END_TIME = int(EVENT_TIME + 32)
PLOT_START = EVENT_TIME - 0.5
PLOT_END = EVENT_TIME + 0.5
BANDPASS_LOW = 30
BANDPASS_HIGH = 300
QTRANSFORM_LOW = 20
QTRANSFORM_HIGH = 400

# --- 1. Data Download ---
print("="*60)
print("Step 1: Downloading strain data for GW170608 (H1 and L1)")
strain_H1 = None
strain_L1 = None
try:
    print(f"  Downloading H1 strain data from {START_TIME} to {END_TIME}...")
    strain_H1 = TimeSeries.fetch_open_data('H1', START_TIME, END_TIME)
    print("  H1 strain data downloaded successfully.")
except Exception as e:
    print(f"  ERROR: Failed to download H1 data: {e}")

try:
    print(f"  Downloading L1 strain data from {START_TIME} to {END_TIME}...")
    strain_L1 = TimeSeries.fetch_open_data('L1', START_TIME, END_TIME)
    print("  L1 strain data downloaded successfully.")
except Exception as e:
    print(f"  ERROR: Failed to download L1 data: {e}")

if strain_H1 is None and strain_L1 is None:
    print("  ERROR: No strain data downloaded. Exiting.")
    exit(1)

# --- 2. Data Processing ---
print("="*60)
print("Step 2: Bandpass filtering and whitening")
strain_H1_proc = None
strain_L1_proc = None

# H1 processing
if strain_H1 is not None:
    try:
        print("  Applying bandpass filter (30–300 Hz) to H1...")
        strain_H1_bp = strain_H1.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
        print("  Whitening H1 data...")
        strain_H1_proc = strain_H1_bp.whiten()
        print("  H1 data processed successfully.")
    except Exception as e:
        print(f"  ERROR: Failed to process H1 data: {e}")
else:
    print("  WARNING: H1 data not available for processing.")

# L1 processing
if strain_L1 is not None:
    try:
        print("  Applying bandpass filter (30–300 Hz) to L1...")
        strain_L1_bp = strain_L1.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
        print("  Whitening L1 data...")
        strain_L1_proc = strain_L1_bp.whiten()
        print("  L1 data processed successfully.")
    except Exception as e:
        print(f"  ERROR: Failed to process L1 data: {e}")
else:
    print("  WARNING: L1 data not available for processing.")

if strain_H1_proc is None and strain_L1_proc is None:
    print("  ERROR: No processed data available. Exiting.")
    exit(1)

# --- 3. Waveform Visualization ---
print("="*60)
print("Step 3: Plotting filtered, whitened strain (±0.5s around event)")
strain_H1_zoom = None
strain_L1_zoom = None

try:
    if strain_H1_proc is not None:
        print(f"  Slicing H1 data to [{PLOT_START}, {PLOT_END}]...")
        strain_H1_zoom = strain_H1_proc.crop(PLOT_START, PLOT_END)
        print("  H1 data sliced.")
    if strain_L1_proc is not None:
        print(f"  Slicing L1 data to [{PLOT_START}, {PLOT_END}]...")
        strain_L1_zoom = strain_L1_proc.crop(PLOT_START, PLOT_END)
        print("  L1 data sliced.")
except Exception as e:
    print(f"  ERROR: Failed to slice data: {e}")

if strain_H1_zoom is not None or strain_L1_zoom is not None:
    try:
        print("  Plotting filtered, whitened strain data...")
        plt.figure(figsize=(10, 6))
        if strain_H1_zoom is not None:
            t_H1 = strain_H1_zoom.times.value - EVENT_TIME
            plt.plot(t_H1, strain_H1_zoom.value, label='H1', color='C0')
        if strain_L1_zoom is not None:
            t_L1 = strain_L1_zoom.times.value - EVENT_TIME
            plt.plot(t_L1, strain_L1_zoom.value, label='L1', color='C1')
        plt.xlabel('Time (s) relative to event')
        plt.ylabel('Whitened strain')
        plt.title('GW170608: Whitened, Bandpassed Strain (±0.5s around event)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig('GW170608_whitened_bandpassed_strain_zoom.png')
        print("  Plot displayed and saved as 'GW170608_whitened_bandpassed_strain_zoom.png'.")
    except Exception as e:
        print(f"  ERROR: Failed to plot strain data: {e}")
else:
    print("  WARNING: No sliced data available for plotting.")

# --- 4. Q-transform Spectrogram Visualization ---
print("="*60)
print("Step 4: Q-transform spectrograms (20–400 Hz, ±0.5s around event)")

def plot_qtransform(strain, detector, event_time, plot_start, plot_end, q_low, q_high):
    try:
        print(f"  Slicing {detector} data to [{plot_start}, {plot_end}]...")
        strain_zoom = strain.crop(plot_start, plot_end)
        print(f"  Computing Q-transform for {detector}...")
        q = strain_zoom.q_transform(frange=(q_low, q_high))
        print(f"  Plotting Q-transform spectrogram for {detector}...")
        fig = q.plot(figsize=(10, 6), vmin=0, vmax=15)
        ax = fig.gca()
        ax.set_title(f'{detector} Q-transform Spectrogram (GW170608, ±0.5s)')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s] relative to event')
        # Set x-axis to be relative to event time
        ax.set_xlim(plot_start - event_time, plot_end - event_time)
        # Overlay event time
        ax.axvline(0, color='r', linestyle='--', label='Event time')
        ax.legend()
        plt.tight_layout()
        plt.show()
        filename = f'GW170608_{detector}_qtransform_zoom.png'
        fig.savefig(filename)
        print(f"  Q-transform spectrogram for {detector} saved as '{filename}'.")
    except Exception as e:
        print(f"  ERROR: Failed Q-transform for {detector}: {e}")

if strain_H1_proc is not None:
    plot_qtransform(strain_H1_proc, 'H1', EVENT_TIME, PLOT_START, PLOT_END, QTRANSFORM_LOW, QTRANSFORM_HIGH)
else:
    print("  WARNING: Processed H1 data not available for Q-transform.")

if strain_L1_proc is not None:
    plot_qtransform(strain_L1_proc, 'L1', EVENT_TIME, PLOT_START, PLOT_END, QTRANSFORM_LOW, QTRANSFORM_HIGH)
else:
    print("  WARNING: Processed L1 data not available for Q-transform.")

print("="*60)
print("Workflow complete.")