spectrogram(stride, fftlength=..., overlap=...)
spectrogram(fftlength, overlap=...)
# ===========================
# GW150914 Gravitational Wave Analysis Script
# ===========================

# ---- Imports ----
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- Parameters ----
gps_event = 1126259462
delta_t = 2048
start = gps_event - delta_t
end = gps_event + delta_t

# Output directory
output_dir = "gw150914_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# ---- 1. Data Loading ----
print("="*60)
print("Step 1: Downloading H1 and L1 strain data from GWOSC...")
strain_H1 = None
strain_L1 = None

try:
    print(f"Fetching H1 strain data from {start} to {end}...")
    strain_H1 = TimeSeries.fetch_open_data('H1', start, end, cache=True)
    print("H1 data fetched successfully.")
    # Save raw H1 data to file
    h1_raw_path = os.path.join(output_dir, "H1_raw_strain.npy")
    np.save(h1_raw_path, strain_H1.value)
    print(f"H1 raw strain data saved to {h1_raw_path}")
except Exception as e:
    print(f"Error fetching H1 data: {e}")

try:
    print(f"Fetching L1 strain data from {start} to {end}...")
    strain_L1 = TimeSeries.fetch_open_data('L1', start, end, cache=True)
    print("L1 data fetched successfully.")
    # Save raw L1 data to file
    l1_raw_path = os.path.join(output_dir, "L1_raw_strain.npy")
    np.save(l1_raw_path, strain_L1.value)
    print(f"L1 raw strain data saved to {l1_raw_path}")
except Exception as e:
    print(f"Error fetching L1 data: {e}")

if strain_H1 is None or strain_L1 is None:
    print("Critical error: Could not fetch both H1 and L1 data. Exiting.")
    exit(1)

# ---- 2. Data Processing ----
print("="*60)
print("Step 2: Whitening and filtering strain data...")

strain_H1_proc = None
strain_L1_proc = None

try:
    print("Whitening H1 data...")
    strain_H1_white = strain_H1.whiten()
    print("Applying highpass filter at 30 Hz to H1 data...")
    strain_H1_hp = strain_H1_white.highpass(30)
    print("Applying lowpass filter at 250 Hz to H1 data...")
    strain_H1_proc = strain_H1_hp.lowpass(250)
    print("H1 data processing complete.")
    # Save processed H1 data
    h1_proc_path = os.path.join(output_dir, "H1_processed_strain.npy")
    np.save(h1_proc_path, strain_H1_proc.value)
    print(f"H1 processed strain data saved to {h1_proc_path}")
except Exception as e:
    print(f"Error processing H1 data: {e}")

try:
    print("Whitening L1 data...")
    strain_L1_white = strain_L1.whiten()
    print("Applying highpass filter at 30 Hz to L1 data...")
    strain_L1_hp = strain_L1_white.highpass(30)
    print("Applying lowpass filter at 250 Hz to L1 data...")
    strain_L1_proc = strain_L1_hp.lowpass(250)
    print("L1 data processing complete.")
    # Save processed L1 data
    l1_proc_path = os.path.join(output_dir, "L1_processed_strain.npy")
    np.save(l1_proc_path, strain_L1_proc.value)
    print(f"L1 processed strain data saved to {l1_proc_path}")
except Exception as e:
    print(f"Error processing L1 data: {e}")

if strain_H1_proc is None or strain_L1_proc is None:
    print("Critical error: Could not process both H1 and L1 data. Exiting.")
    exit(1)

# ---- 3. Time-Domain Visualization ----
print("="*60)
print("Step 3: Creating time-domain plots around the merger...")

window = 0.2  # seconds
plot_start = gps_event - window
plot_end = gps_event + window

# H1 time-domain plot
try:
    print("Preparing H1 time-domain plot...")
    h1_zoom = strain_H1_proc.crop(plot_start, plot_end)
    plt.figure(figsize=(10, 4))
    plt.plot(h1_zoom.times.value, h1_zoom.value, label='H1')
    plt.title('H1 Strain Data (Whitened & Filtered)\nGW150914, GPS {:.1f} ± {:.1f}s'.format(gps_event, window))
    plt.xlabel('Time (s) since GPS {}'.format(gps_event))
    plt.ylabel('Strain')
    plt.xlim(plot_start, plot_end)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    h1_time_plot_path = os.path.join(output_dir, "H1_time_domain.png")
    plt.savefig(h1_time_plot_path)
    plt.show()
    print(f"H1 time-domain plot saved to {h1_time_plot_path}")
except Exception as e:
    print(f"Error plotting H1 data: {e}")

# L1 time-domain plot
try:
    print("Preparing L1 time-domain plot...")
    l1_zoom = strain_L1_proc.crop(plot_start, plot_end)
    plt.figure(figsize=(10, 4))
    plt.plot(l1_zoom.times.value, l1_zoom.value, label='L1', color='orange')
    plt.title('L1 Strain Data (Whitened & Filtered)\nGW150914, GPS {:.1f} ± {:.1f}s'.format(gps_event, window))
    plt.xlabel('Time (s) since GPS {}'.format(gps_event))
    plt.ylabel('Strain')
    plt.xlim(plot_start, plot_end)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    l1_time_plot_path = os.path.join(output_dir, "L1_time_domain.png")
    plt.savefig(l1_time_plot_path)
    plt.show()
    print(f"L1 time-domain plot saved to {l1_time_plot_path}")
except Exception as e:
    print(f"Error plotting L1 data: {e}")

# ---- 4. Time-Frequency Spectrogram Visualization ----
print("="*60)
print("Step 4: Generating time-frequency spectrograms...")

spec_start = gps_event - window
spec_end = gps_event + window
fftlength = 0.05  # seconds
overlap = 0.025   # seconds

# H1 spectrogram
try:
    print("Generating H1 spectrogram...")
    h1_zoom = strain_H1_proc.crop(spec_start, spec_end)
    # FIX: Provide fftlength as the first positional argument, overlap as keyword
    h1_spec = h1_zoom.spectrogram(fftlength, overlap=overlap)
    plt.figure(figsize=(10, 5))
    h1_spec.plot(norm='log', vmin=1e-24, vmax=1e-21)
    plt.title('H1 Time-Frequency Spectrogram\nGW150914, GPS {:.1f} ± {:.1f}s'.format(gps_event, window))
    plt.xlabel('Time (s) since GPS {}'.format(gps_event))
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    h1_spec_plot_path = os.path.join(output_dir, "H1_spectrogram.png")
    plt.savefig(h1_spec_plot_path)
    plt.show()
    print(f"H1 spectrogram saved to {h1_spec_plot_path}")
except Exception as e:
    print(f"Error generating H1 spectrogram: {e}")

# L1 spectrogram
try:
    print("Generating L1 spectrogram...")
    l1_zoom = strain_L1_proc.crop(spec_start, spec_end)
    # FIX: Provide fftlength as the first positional argument, overlap as keyword
    l1_spec = l1_zoom.spectrogram(fftlength, overlap=overlap)
    plt.figure(figsize=(10, 5))
    l1_spec.plot(norm='log', vmin=1e-24, vmax=1e-21)
    plt.title('L1 Time-Frequency Spectrogram\nGW150914, GPS {:.1f} ± {:.1f}s'.format(gps_event, window))
    plt.xlabel('Time (s) since GPS {}'.format(gps_event))
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    l1_spec_plot_path = os.path.join(output_dir, "L1_spectrogram.png")
    plt.savefig(l1_spec_plot_path)
    plt.show()
    print(f"L1 spectrogram saved to {l1_spec_plot_path}")
except Exception as e:
    print(f"Error generating L1 spectrogram: {e}")

print("="*60)
print("Analysis complete. All outputs saved in:", output_dir)