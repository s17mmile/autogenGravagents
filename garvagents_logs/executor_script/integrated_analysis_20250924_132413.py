# --- Imports ---
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# --- Parameters ---
gps_center = 1180922494.5
delta = 32
start_time = int(gps_center - delta)  # 1180922462
end_time = int(gps_center + delta)    # 1180922526

zoom_window = 0.5  # seconds
zoom_start = gps_center - zoom_window  # 1180922494.0
zoom_end = gps_center + zoom_window    # 1180922495.0

bandpass_low = 30
bandpass_high = 300

qtrans_fmin = 20
qtrans_fmax = 400

# --- 1. Download Strain Data ---
print("="*60)
print("Step 1: Downloading strain data for GW170608")
print(f"Time window: {start_time} to {end_time} (GPS seconds)")

strain_H1 = None
strain_L1 = None

try:
    print("Fetching H1 strain data...")
    strain_H1 = TimeSeries.fetch_open_data('H1', start_time, end_time, cache=True)
    print("H1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error downloading H1 data: {e}")

try:
    print("Fetching L1 strain data...")
    strain_L1 = TimeSeries.fetch_open_data('L1', start_time, end_time, cache=True)
    print("L1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error downloading L1 data: {e}")

if strain_H1 is None or strain_L1 is None:
    print("ERROR: Could not download both H1 and L1 data. Exiting.")
    exit(1)

# --- 2. Processing: Bandpass and Whiten ---
print("="*60)
print("Step 2: Processing strain data (bandpass and whiten)")

strain_H1_proc = None
strain_L1_proc = None

try:
    print(f"Applying bandpass filter ({bandpass_low}-{bandpass_high} Hz) to H1 data...")
    strain_H1_bp = strain_H1.bandpass(bandpass_low, bandpass_high)
    print("Whitening H1 data...")
    strain_H1_proc = strain_H1_bp.whiten()
    print("H1 data filtered and whitened successfully.")
except Exception as e:
    print(f"Error processing H1 data: {e}")

try:
    print(f"Applying bandpass filter ({bandpass_low}-{bandpass_high} Hz) to L1 data...")
    strain_L1_bp = strain_L1.bandpass(bandpass_low, bandpass_high)
    print("Whitening L1 data...")
    strain_L1_proc = strain_L1_bp.whiten()
    print("L1 data filtered and whitened successfully.")
except Exception as e:
    print(f"Error processing L1 data: {e}")

if strain_H1_proc is None or strain_L1_proc is None:
    print("ERROR: Could not process both H1 and L1 data. Exiting.")
    exit(1)

# --- 3. Visualization: Time Series (±0.5s) ---
print("="*60)
print("Step 3: Visualizing filtered, whitened strain (±0.5s)")

# Slice the processed data
try:
    print(f"Slicing H1 data to [{zoom_start}, {zoom_end}]...")
    strain_H1_zoom = strain_H1_proc.crop(zoom_start, zoom_end)
    print("H1 data sliced successfully.")
except Exception as e:
    print(f"Error slicing H1 data: {e}")
    strain_H1_zoom = None

try:
    print(f"Slicing L1 data to [{zoom_start}, {zoom_end}]...")
    strain_L1_zoom = strain_L1_proc.crop(zoom_start, zoom_end)
    print("L1 data sliced successfully.")
except Exception as e:
    print(f"Error slicing L1 data: {e}")
    strain_L1_zoom = None

# Plot both detectors on the same figure
try:
    print("Plotting H1 and L1 data (combined)...")
    plt.figure(figsize=(10, 6))
    if strain_H1_zoom is not None:
        plt.plot(strain_H1_zoom.times.value, strain_H1_zoom.value, label='H1')
    if strain_L1_zoom is not None:
        plt.plot(strain_L1_zoom.times.value, strain_L1_zoom.value, label='L1')
    plt.xlabel('GPS Time (s)')
    plt.ylabel('Whitened Strain')
    plt.title('GW170608: Whitened, Bandpassed Strain (±0.5s around event)')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.savefig('GW170608_whitened_strain_H1_L1.png')
    print("Combined plot saved as 'GW170608_whitened_strain_H1_L1.png'.")
except Exception as e:
    print(f"Error during combined plotting: {e}")

# Individual plots
try:
    if strain_H1_zoom is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(strain_H1_zoom.times.value, strain_H1_zoom.value, color='b')
        plt.xlabel('GPS Time (s)')
        plt.ylabel('Whitened Strain')
        plt.title('GW170608: H1 Whitened, Bandpassed Strain (±0.5s)')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.savefig('GW170608_whitened_strain_H1.png')
        print("H1 plot saved as 'GW170608_whitened_strain_H1.png'.")
except Exception as e:
    print(f"Error during H1 plotting: {e}")

try:
    if strain_L1_zoom is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(strain_L1_zoom.times.value, strain_L1_zoom.value, color='g')
        plt.xlabel('GPS Time (s)')
        plt.ylabel('Whitened Strain')
        plt.title('GW170608: L1 Whitened, Bandpassed Strain (±0.5s)')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.savefig('GW170608_whitened_strain_L1.png')
        print("L1 plot saved as 'GW170608_whitened_strain_L1.png'.")
except Exception as e:
    print(f"Error during L1 plotting: {e}")

# --- 4. Visualization: Q-transform Spectrograms ---
print("="*60)
print("Step 4: Q-transform spectrograms (20–400 Hz, ±0.5s)")

# To avoid "window is longer than input signal", set qrange and/or segment length
# Use a slightly longer segment for Q-transform (e.g., ±2s)
qtransform_window = 2.0  # seconds
qtransform_start = gps_center - qtransform_window
qtransform_end = gps_center + qtransform_window

# Crop a longer segment for Q-transform
try:
    print(f"Slicing H1 data for Q-transform to [{qtransform_start}, {qtransform_end}]...")
    strain_H1_qt = strain_H1_proc.crop(qtransform_start, qtransform_end)
    print("H1 data for Q-transform sliced successfully.")
except Exception as e:
    print(f"Error slicing H1 data for Q-transform: {e}")
    strain_H1_qt = None

try:
    print(f"Slicing L1 data for Q-transform to [{qtransform_start}, {qtransform_end}]...")
    strain_L1_qt = strain_L1_proc.crop(qtransform_start, qtransform_end)
    print("L1 data for Q-transform sliced successfully.")
except Exception as e:
    print(f"Error slicing L1 data for Q-transform: {e}")
    strain_L1_qt = None

# H1 Q-transform
try:
    if strain_H1_qt is not None:
        print("Computing Q-transform for H1...")
        q_h1 = strain_H1_qt.q_transform(outseg=(qtransform_start, qtransform_end))
        print("Plotting Q-transform for H1...")
        fig_h1 = q_h1.plot(figsize=(10, 6), vmin=0, vmax=20)
        ax_h1 = fig_h1.gca()
        ax_h1.set_ylim(qtrans_fmin, qtrans_fmax)
        ax_h1.set_title("GW170608: H1 Q-transform (20–400 Hz, ±2s)")
        plt.tight_layout()
        plt.show()
        fig_h1.savefig("GW170608_qtransform_H1.png")
        print("H1 Q-transform plot saved as 'GW170608_qtransform_H1.png'.")
    else:
        print("H1 Q-transform skipped due to missing data segment.")
except Exception as e:
    print(f"Error computing or plotting H1 Q-transform: {e}")

# L1 Q-transform
try:
    if strain_L1_qt is not None:
        print("Computing Q-transform for L1...")
        q_l1 = strain_L1_qt.q_transform(outseg=(qtransform_start, qtransform_end))
        print("Plotting Q-transform for L1...")
        fig_l1 = q_l1.plot(figsize=(10, 6), vmin=0, vmax=20)
        ax_l1 = fig_l1.gca()
        ax_l1.set_ylim(qtrans_fmin, qtrans_fmax)
        ax_l1.set_title("GW170608: L1 Q-transform (20–400 Hz, ±2s)")
        plt.tight_layout()
        plt.show()
        fig_l1.savefig("GW170608_qtransform_L1.png")
        print("L1 Q-transform plot saved as 'GW170608_qtransform_L1.png'.")
    else:
        print("L1 Q-transform skipped due to missing data segment.")
except Exception as e:
    print(f"Error computing or plotting L1 Q-transform: {e}")

print("="*60)
print("Workflow complete. All plots displayed and saved.")