# --- Imports ---
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# --- Constants ---
GPS_CENTER = 1180922494.5
DURATION = 64  # seconds
HALF_DURATION = DURATION / 2
GPS_START = GPS_CENTER - HALF_DURATION
GPS_END = GPS_CENTER + HALF_DURATION
BANDPASS_LOW = 30
BANDPASS_HIGH = 300
QTRANSFORM_FRANGE = (20, 400)
ZOOM_WINDOW = 0.5  # seconds

# --- Data Download ---
print("="*60)
print("Step 1: Downloading LIGO strain data for GW170608")
print(f"  GPS start: {GPS_START}")
print(f"  GPS end:   {GPS_END}")
print(f"  Duration:  {DURATION} seconds")

h1_strain = None
l1_strain = None

# Download H1 data
try:
    print("Fetching H1 strain data...")
    h1_strain = TimeSeries.fetch_open_data('H1', GPS_START, GPS_END)
    print("H1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error fetching H1 data: {e}")

# Download L1 data
try:
    print("Fetching L1 strain data...")
    l1_strain = TimeSeries.fetch_open_data('L1', GPS_START, GPS_END)
    print("L1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error fetching L1 data: {e}")

if h1_strain is None and l1_strain is None:
    print("ERROR: No strain data downloaded. Exiting.")
    exit(1)

# --- Preprocessing ---
print("="*60)
print("Step 2: Preprocessing: Bandpass filtering (30-300 Hz) and whitening.")

h1_strain_processed = None
l1_strain_processed = None

# Process H1 data
if h1_strain is not None:
    try:
        print("Processing H1 strain data...")
        h1_strain_processed = h1_strain.bandpass(BANDPASS_LOW, BANDPASS_HIGH).whiten()
        print("H1 strain data processed successfully.")
    except Exception as e:
        print(f"Error processing H1 strain data: {e}")
else:
    print("H1 strain data not found. Skipping H1 processing.")

# Process L1 data
if l1_strain is not None:
    try:
        print("Processing L1 strain data...")
        l1_strain_processed = l1_strain.bandpass(BANDPASS_LOW, BANDPASS_HIGH).whiten()
        print("L1 strain data processed successfully.")
    except Exception as e:
        print(f"Error processing L1 strain data: {e}")
else:
    print("L1 strain data not found. Skipping L1 processing.")

if h1_strain_processed is None and l1_strain_processed is None:
    print("ERROR: No processed strain data available. Exiting.")
    exit(1)

# --- Visualization ---
print("="*60)
print("Step 3: Visualization: Time-domain and Q-transform plots.")

def plot_detector(detector_name, strain_processed):
    if strain_processed is None:
        print(f"No processed data for {detector_name}, skipping visualization.")
        return

    print(f"Creating plots for {detector_name}...")

    # Time crop for ±0.5s around event
    try:
        strain_zoom = strain_processed.crop(GPS_CENTER - ZOOM_WINDOW, GPS_CENTER + ZOOM_WINDOW)
    except Exception as e:
        print(f"Error cropping {detector_name} data: {e}")
        return

    # Q-transform
    try:
        print(f"Computing Q-transform for {detector_name}...")
        q = strain_processed.q_transform(frange=QTRANSFORM_FRANGE)
        q_zoom = q.crop(GPS_CENTER - ZOOM_WINDOW, GPS_CENTER + ZOOM_WINDOW)
    except Exception as e:
        print(f"Error computing Q-transform for {detector_name}: {e}")
        return

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                            gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle(f"{detector_name} Strain Data around GW170608")

    # Time-domain plot
    axs[0].plot(strain_zoom.times.value - GPS_CENTER, strain_zoom.value, color='black', lw=0.8)
    axs[0].set_ylabel("Whitened Strain")
    axs[0].set_xlim(-ZOOM_WINDOW, ZOOM_WINDOW)
    axs[0].set_title("Time-domain (±0.5 s)")

    # Q-transform spectrogram
    qax = q_zoom.plot(ax=axs[1], vmin=0, vmax=15, cmap='viridis')
    axs[1].set_ylabel("Frequency [Hz]")
    axs[1].set_xlabel("Time [s] relative to event")
    axs[1].set_title("Q-transform Spectrogram (20–400 Hz, ±0.5 s)")
    axs[1].set_xlim(-ZOOM_WINDOW, ZOOM_WINDOW)

    # Adjust colorbar for Q-transform
    plt.colorbar(qax, ax=axs[1], label='Normalized energy')

    # Save and show
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"{detector_name}_GW170608_zoomed_plots.png"
    plt.savefig(filename)
    print(f"Saved {detector_name} plot to {filename}")
    plt.show()

# Visualize for H1
if h1_strain_processed is not None:
    plot_detector('H1', h1_strain_processed)
else:
    print("No processed H1 data found for visualization.")

# Visualize for L1
if l1_strain_processed is not None:
    plot_detector('L1', l1_strain_processed)
else:
    print("No processed L1 data found for visualization.")

print("="*60)
print("All tasks completed.")