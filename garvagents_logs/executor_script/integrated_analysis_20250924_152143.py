# --- Imports ---
import os
import sys
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# --- Parameters and Output Paths ---
event_gps = 1180922494.5
duration = 4  # seconds
half_duration = duration / 2
start_time = event_gps - half_duration
end_time = event_gps + half_duration
detectors = ['H1', 'L1']

# Bandpass filter parameters
low_freq = 35
high_freq = 350

# Time-domain plot window (seconds)
plot_window = 0.2

# Q-transform parameters
q_window = 0.5  # seconds before and after merger
q_low_freq = 30
q_high_freq = 400

# Output directory
output_dir = "gw170608_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Task 1: Download Strain Data ---
print("="*60)
print("TASK 1: Downloading strain data for GW170608")
print("="*60)
strain_H1 = None
strain_L1 = None

print(f"Fetching {duration} seconds of strain data for GW170608 (GPS {event_gps})...")
print(f"Time window: {start_time} to {end_time}")

try:
    print("Downloading H1 data...")
    strain_H1 = TimeSeries.fetch_open_data('H1', start_time, end_time, cache=True)
    print("H1 data downloaded successfully.")
    strain_H1.write(os.path.join(output_dir, "strain_H1.gwf"), format='gwf')
except Exception as e:
    print(f"Error downloading H1 data: {e}")

try:
    print("Downloading L1 data...")
    strain_L1 = TimeSeries.fetch_open_data('L1', start_time, end_time, cache=True)
    print("L1 data downloaded successfully.")
    strain_L1.write(os.path.join(output_dir, "strain_L1.gwf"), format='gwf')
except Exception as e:
    print(f"Error downloading L1 data: {e}")

if strain_H1 is None and strain_L1 is None:
    print("ERROR: No strain data could be downloaded. Exiting.")
    sys.exit(1)

# --- Task 2: Bandpass Filtering ---
print("\n" + "="*60)
print("TASK 2: Applying bandpass filter ({}-{} Hz)".format(low_freq, high_freq))
print("="*60)
filtered_H1 = None
filtered_L1 = None

try:
    if strain_H1 is not None:
        print("Filtering H1 data...")
        filtered_H1 = strain_H1.bandpass(low_freq, high_freq)
        print("H1 data filtered successfully.")
        filtered_H1.write(os.path.join(output_dir, "filtered_H1.gwf"), format='gwf')
    else:
        print("H1 data not available for filtering.")
except Exception as e:
    print(f"Error filtering H1 data: {e}")

try:
    if strain_L1 is not None:
        print("Filtering L1 data...")
        filtered_L1 = strain_L1.bandpass(low_freq, high_freq)
        print("L1 data filtered successfully.")
        filtered_L1.write(os.path.join(output_dir, "filtered_L1.gwf"), format='gwf')
    else:
        print("L1 data not available for filtering.")
except Exception as e:
    print(f"Error filtering L1 data: {e}")

if filtered_H1 is None and filtered_L1 is None:
    print("ERROR: No filtered data available. Exiting.")
    sys.exit(1)

# --- Task 3: Time-Domain Plot ---
print("\n" + "="*60)
print("TASK 3: Plotting time-domain strain data (±{} s)".format(plot_window))
print("="*60)
cropped_H1 = None
cropped_L1 = None

try:
    if filtered_H1 is not None:
        print("Cropping H1 data to ±{} seconds around the merger...".format(plot_window))
        cropped_H1 = filtered_H1.crop(event_gps - plot_window, event_gps + plot_window)
        print("H1 data cropped successfully.")
        cropped_H1.write(os.path.join(output_dir, "cropped_H1.gwf"), format='gwf')
    else:
        print("Filtered H1 data not available for cropping.")
except Exception as e:
    print(f"Error cropping H1 data: {e}")

try:
    if filtered_L1 is not None:
        print("Cropping L1 data to ±{} seconds around the merger...".format(plot_window))
        cropped_L1 = filtered_L1.crop(event_gps - plot_window, event_gps + plot_window)
        print("L1 data cropped successfully.")
        cropped_L1.write(os.path.join(output_dir, "cropped_L1.gwf"), format='gwf')
    else:
        print("Filtered L1 data not available for cropping.")
except Exception as e:
    print(f"Error cropping L1 data: {e}")

# Plotting
try:
    print("Plotting time-domain strain data for H1 and L1...")
    plt.figure(figsize=(10, 6))
    plotted = False
    if cropped_H1 is not None:
        plt.plot(
            cropped_H1.times.value - event_gps,
            cropped_H1.value,
            label='H1',
            color='C0'
        )
        plotted = True
    if cropped_L1 is not None:
        plt.plot(
            cropped_L1.times.value - event_gps,
            cropped_L1.value,
            label='L1',
            color='C1'
        )
        plotted = True
    if plotted:
        plt.xlabel('Time (s) relative to merger')
        plt.ylabel('Strain')
        plt.title('Filtered Strain Data around GW170608 Merger (±0.2 s)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "time_domain_strain.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"Time-domain plot saved to {plot_path}")
    else:
        print("No cropped data available for plotting.")
except Exception as e:
    print(f"Error during plotting: {e}")

# --- Task 4: Q-Transform Spectrograms ---
print("\n" + "="*60)
print("TASK 4: Generating Q-transform spectrograms ({}-{} Hz, ±{} s)".format(q_low_freq, q_high_freq, q_window))
print("="*60)
q_H1 = None
q_L1 = None

try:
    if filtered_H1 is not None:
        print("Cropping H1 data for Q-transform...")
        q_H1 = filtered_H1.crop(event_gps - q_window, event_gps + q_window)
        print("H1 data cropped for Q-transform.")
        q_H1.write(os.path.join(output_dir, "q_H1.gwf"), format='gwf')
    else:
        print("Filtered H1 data not available for Q-transform cropping.")
except Exception as e:
    print(f"Error cropping H1 data for Q-transform: {e}")

try:
    if filtered_L1 is not None:
        print("Cropping L1 data for Q-transform...")
        q_L1 = filtered_L1.crop(event_gps - q_window, event_gps + q_window)
        print("L1 data cropped for Q-transform.")
        q_L1.write(os.path.join(output_dir, "q_L1.gwf"), format='gwf')
    else:
        print("Filtered L1 data not available for Q-transform cropping.")
except Exception as e:
    print(f"Error cropping L1 data for Q-transform: {e}")

# Generate and plot Q-transform spectrograms
try:
    if q_H1 is not None:
        print("Generating Q-transform for H1...")
        spec_H1 = q_H1.q_transform(outseg=(event_gps - q_window, event_gps + q_window), 
                                   frange=(q_low_freq, q_high_freq))
        print("Plotting H1 Q-transform spectrogram...")
        ax1 = spec_H1.plot(figsize=(10, 4))
        ax1.set_title("GW170608 H1 Q-transform Spectrogram")
        ax1.set_epoch(event_gps)
        ax1.set_ylabel("Frequency [Hz]")
        ax1.set_xlabel("Time [s] relative to merger")
        q_plot_path = os.path.join(output_dir, "q_transform_H1.png")
        plt.savefig(q_plot_path)
        plt.show()
        print(f"H1 Q-transform spectrogram saved to {q_plot_path}")
    else:
        print("H1 Q-transform skipped due to missing data.")
except Exception as e:
    print(f"Error generating or plotting H1 Q-transform: {e}")

try:
    if q_L1 is not None:
        print("Generating Q-transform for L1...")
        spec_L1 = q_L1.q_transform(outseg=(event_gps - q_window, event_gps + q_window), 
                                   frange=(q_low_freq, q_high_freq))
        print("Plotting L1 Q-transform spectrogram...")
        ax2 = spec_L1.plot(figsize=(10, 4))
        ax2.set_title("GW170608 L1 Q-transform Spectrogram")
        ax2.set_epoch(event_gps)
        ax2.set_ylabel("Frequency [Hz]")
        ax2.set_xlabel("Time [s] relative to merger")
        q_plot_path = os.path.join(output_dir, "q_transform_L1.png")
        plt.savefig(q_plot_path)
        plt.show()
        print(f"L1 Q-transform spectrogram saved to {q_plot_path}")
    else:
        print("L1 Q-transform skipped due to missing data.")
except Exception as e:
    print(f"Error generating or plotting L1 Q-transform: {e}")

print("\nAll tasks completed. Results saved in '{}'.".format(output_dir))