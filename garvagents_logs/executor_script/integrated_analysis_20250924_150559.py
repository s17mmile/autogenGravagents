# --- Imports ---
import os
import sys
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# --- Parameters ---
EVENT_GPS = 1180922494.5
DELTA_T = 32  # seconds before and after event
START_TIME = EVENT_GPS - DELTA_T
END_TIME = EVENT_GPS + DELTA_T
LOW_FREQ = 35
HIGH_FREQ = 350
CROP_WINDOW = 0.2  # seconds for plotting around event
OUTPUT_DIR = "gw170608_analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Download Strain Data ---
print("="*60)
print("Step 1: Downloading H1 and L1 strain data for GW170608...")
strain_H1 = None
strain_L1 = None
try:
    print(f"Fetching H1 strain data from {START_TIME} to {END_TIME} (GPS seconds)...")
    strain_H1 = TimeSeries.fetch_open_data('H1', START_TIME, END_TIME, cache=True)
    print("H1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error downloading H1 data: {e}")
    sys.exit(1)

try:
    print(f"Fetching L1 strain data from {START_TIME} to {END_TIME} (GPS seconds)...")
    strain_L1 = TimeSeries.fetch_open_data('L1', START_TIME, END_TIME, cache=True)
    print("L1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error downloading L1 data: {e}")
    sys.exit(1)

# Optionally save raw data
try:
    strain_H1.write(os.path.join(OUTPUT_DIR, "strain_H1_raw.gwf"), format='gwf')
    strain_L1.write(os.path.join(OUTPUT_DIR, "strain_L1_raw.gwf"), format='gwf')
    print("Raw strain data saved.")
except Exception as e:
    print(f"Warning: Could not save raw strain data: {e}")

# --- 2. Bandpass Filtering ---
print("="*60)
print(f"Step 2: Applying {LOW_FREQ}-{HIGH_FREQ} Hz bandpass filter to strain data...")
strain_H1_bp = None
strain_L1_bp = None
try:
    print("Filtering H1 strain data...")
    strain_H1_bp = strain_H1.bandpass(LOW_FREQ, HIGH_FREQ)
    print("H1 strain data filtered successfully.")
except Exception as e:
    print(f"Error filtering H1 data: {e}")
    sys.exit(1)

try:
    print("Filtering L1 strain data...")
    strain_L1_bp = strain_L1.bandpass(LOW_FREQ, HIGH_FREQ)
    print("L1 strain data filtered successfully.")
except Exception as e:
    print(f"Error filtering L1 data: {e}")
    sys.exit(1)

# Optionally save filtered data
try:
    strain_H1_bp.write(os.path.join(OUTPUT_DIR, "strain_H1_bandpassed.gwf"), format='gwf')
    strain_L1_bp.write(os.path.join(OUTPUT_DIR, "strain_L1_bandpassed.gwf"), format='gwf')
    print("Filtered strain data saved.")
except Exception as e:
    print(f"Warning: Could not save filtered strain data: {e}")

# --- 3. Plot Filtered Time Series ---
print("="*60)
print("Step 3: Plotting filtered strain time series for H1 and L1...")
try:
    t0 = EVENT_GPS - CROP_WINDOW
    t1 = EVENT_GPS + CROP_WINDOW
    strain_H1_crop = strain_H1_bp.crop(t0, t1)
    strain_L1_crop = strain_L1_bp.crop(t0, t1)
    time_H1 = strain_H1_crop.times.value - EVENT_GPS
    time_L1 = strain_L1_crop.times.value - EVENT_GPS

    plt.figure(figsize=(10, 6))
    plt.plot(time_H1, strain_H1_crop.value, label='H1', color='C0')
    plt.plot(time_L1, strain_L1_crop.value, label='L1', color='C1')
    plt.xlabel('Time (s) relative to GW170608')
    plt.ylabel('Strain')
    plt.title('Filtered Strain Time Series (35–350 Hz)\nGW170608, ±0.2 s around event')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    timeseries_plot_path = os.path.join(OUTPUT_DIR, "filtered_strain_timeseries.png")
    plt.savefig(timeseries_plot_path)
    plt.show()
    print(f"Filtered strain time series plot saved to {timeseries_plot_path}")
except Exception as e:
    print(f"Error during time series plotting: {e}")
    sys.exit(1)

# --- 4. Q-transform Visualization ---
print("="*60)
print("Step 4: Computing and plotting Q-transform for H1 and L1...")
fmin = LOW_FREQ
fmax = HIGH_FREQ
try:
    print("Computing Q-transform for H1...")
    q_H1 = strain_H1_crop.q_transform(frange=(fmin, fmax))
    print("Q-transform for H1 computed.")

    print("Computing Q-transform for L1...")
    q_L1 = strain_L1_crop.q_transform(frange=(fmin, fmax))
    print("Q-transform for L1 computed.")

    # Plot Q-transform for H1
    print("Plotting Q-transform for H1...")
    fig1 = q_H1.plot()
    ax1 = fig1.gca()
    ax1.set_xlim(EVENT_GPS - CROP_WINDOW, EVENT_GPS + CROP_WINDOW)
    ax1.set_ylim(fmin, fmax)
    ax1.set_title("H1 Q-transform\nGW170608, 35–350 Hz, ±0.2 s around event")
    ax1.set_xlabel("Time (GPS s)")
    ax1.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    qtransform_H1_path = os.path.join(OUTPUT_DIR, "qtransform_H1.png")
    plt.savefig(qtransform_H1_path)
    plt.show()
    print(f"H1 Q-transform plot saved to {qtransform_H1_path}")

    # Plot Q-transform for L1
    print("Plotting Q-transform for L1...")
    fig2 = q_L1.plot()
    ax2 = fig2.gca()
    ax2.set_xlim(EVENT_GPS - CROP_WINDOW, EVENT_GPS + CROP_WINDOW)
    ax2.set_ylim(fmin, fmax)
    ax2.set_title("L1 Q-transform\nGW170608, 35–350 Hz, ±0.2 s around event")
    ax2.set_xlabel("Time (GPS s)")
    ax2.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    qtransform_L1_path = os.path.join(OUTPUT_DIR, "qtransform_L1.png")
    plt.savefig(qtransform_L1_path)
    plt.show()
    print(f"L1 Q-transform plot saved to {qtransform_L1_path}")

    print("Q-transform plotting complete.")
except Exception as e:
    print(f"Error during Q-transform computation or plotting: {e}")
    sys.exit(1)

print("="*60)
print("All steps completed successfully. Results saved in:", OUTPUT_DIR)