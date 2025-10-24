# ===========================
# GW150914 Chirp Visualization and Spectrogram Analysis Script
# ===========================

# ---- Imports ----
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- Output Directory ----
output_dir = "gw150914_chirp_outputs"
os.makedirs(output_dir, exist_ok=True)

# ---- 1. Data Loading ----
print("="*60)
print("Step 1: Downloading H1 and L1 strain data from GWOSC...")
gps_center = 1126259462
start = gps_center - 16
end = gps_center + 16

strain_H1 = None
strain_L1 = None

try:
    print("Fetching H1 strain data from GWOSC...")
    strain_H1 = TimeSeries.fetch_open_data('H1', start, end, cache=True)
    print("H1 data download complete.")
    np.save(os.path.join(output_dir, "H1_raw_strain.npy"), strain_H1.value)
except Exception as e:
    print(f"Error fetching H1 data: {e}")

try:
    print("Fetching L1 strain data from GWOSC...")
    strain_L1 = TimeSeries.fetch_open_data('L1', start, end, cache=True)
    print("L1 data download complete.")
    np.save(os.path.join(output_dir, "L1_raw_strain.npy"), strain_L1.value)
except Exception as e:
    print(f"Error fetching L1 data: {e}")

if strain_H1 is None or strain_L1 is None:
    print("Critical error: Could not fetch both H1 and L1 data. Exiting.")
    exit(1)

# ---- 2. Data Processing ----
print("="*60)
print("Step 2: Cropping and bandpass filtering...")

crop_start = 1126259460
crop_end = 1126259464
low_freq = 20
high_freq = 300
filter_order = 4

strain_H1_proc = None
strain_L1_proc = None

try:
    print(f"Cropping H1 data to [{crop_start}, {crop_end}]...")
    h1_cropped = strain_H1.crop(crop_start, crop_end)
    print(f"Applying 4th-order Butterworth bandpass filter ({low_freq}-{high_freq} Hz) to H1...")
    strain_H1_proc = h1_cropped.bandpass(low_freq, high_freq, filtfilt=True, order=filter_order)
    print("H1 processing complete.")
    np.save(os.path.join(output_dir, "H1_bandpassed_strain.npy"), strain_H1_proc.value)
except Exception as e:
    print(f"Error processing H1 data: {e}")

try:
    print(f"Cropping L1 data to [{crop_start}, {crop_end}]...")
    l1_cropped = strain_L1.crop(crop_start, crop_end)
    print(f"Applying 4th-order Butterworth bandpass filter ({low_freq}-{high_freq} Hz) to L1...")
    strain_L1_proc = l1_cropped.bandpass(low_freq, high_freq, filtfilt=True, order=filter_order)
    print("L1 processing complete.")
    np.save(os.path.join(output_dir, "L1_bandpassed_strain.npy"), strain_L1_proc.value)
except Exception as e:
    print(f"Error processing L1 data: {e}")

if strain_H1_proc is None or strain_L1_proc is None:
    print("Critical error: Could not process both H1 and L1 data. Exiting.")
    exit(1)

# ---- 3. Time-Domain Visualization ----
print("="*60)
print("Step 3: Creating time-domain plots (individual and overlaid)...")

zoom_start = 1126259461.5
zoom_end = 1126259462.5

try:
    print("Cropping H1 processed data for time-domain plot...")
    h1_zoom = strain_H1_proc.crop(zoom_start, zoom_end)
    print("Cropping L1 processed data for time-domain plot...")
    l1_zoom = strain_L1_proc.crop(zoom_start, zoom_end)
except Exception as e:
    print(f"Error cropping data for time-domain plots: {e}")
    h1_zoom = None
    l1_zoom = None

# Plot H1 only
try:
    print("Plotting H1 filtered strain (time domain)...")
    plt.figure(figsize=(10, 4))
    plt.plot(h1_zoom.times.value, h1_zoom.value, label='H1', color='C0')
    plt.title('H1 Filtered Strain (GW150914)\nGPS {:.1f} ± 0.5s'.format(gps_center))
    plt.xlabel('Time (s) [GPS]')
    plt.ylabel('Strain')
    plt.grid(True)
    plt.tight_layout()
    h1_time_plot_path = os.path.join(output_dir, "H1_time_domain.png")
    plt.savefig(h1_time_plot_path)
    plt.show()
    print(f"H1 time-domain plot saved to {h1_time_plot_path}")
except Exception as e:
    print(f"Error plotting H1 time-domain data: {e}")

# Plot L1 only
try:
    print("Plotting L1 filtered strain (time domain)...")
    plt.figure(figsize=(10, 4))
    plt.plot(l1_zoom.times.value, l1_zoom.value, label='L1', color='C1')
    plt.title('L1 Filtered Strain (GW150914)\nGPS {:.1f} ± 0.5s'.format(gps_center))
    plt.xlabel('Time (s) [GPS]')
    plt.ylabel('Strain')
    plt.grid(True)
    plt.tight_layout()
    l1_time_plot_path = os.path.join(output_dir, "L1_time_domain.png")
    plt.savefig(l1_time_plot_path)
    plt.show()
    print(f"L1 time-domain plot saved to {l1_time_plot_path}")
except Exception as e:
    print(f"Error plotting L1 time-domain data: {e}")

# Overlay H1 and L1
try:
    print("Plotting overlaid H1 and L1 filtered strain (time domain)...")
    plt.figure(figsize=(10, 4))
    plt.plot(h1_zoom.times.value, h1_zoom.value, label='H1', color='C0')
    plt.plot(l1_zoom.times.value, l1_zoom.value, label='L1', color='C1', alpha=0.7)
    plt.title('H1 and L1 Filtered Strain (GW150914)\nGPS {:.1f} ± 0.5s'.format(gps_center))
    plt.xlabel('Time (s) [GPS]')
    plt.ylabel('Strain')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    overlay_time_plot_path = os.path.join(output_dir, "H1_L1_overlay_time_domain.png")
    plt.savefig(overlay_time_plot_path)
    plt.show()
    print(f"Overlay time-domain plot saved to {overlay_time_plot_path}")
except Exception as e:
    print(f"Error plotting overlaid time-domain data: {e}")

# ---- 4. Spectrogram Visualization ----
print("="*60)
print("Step 4: Generating time-frequency spectrograms (matplotlib and GWpy, raw and whitened)...")

fmin = 20
fmax = 512
NFFT = 256
noverlap = NFFT // 2

def plot_matplotlib_specgram(strain, detector_label, whitened=False):
    try:
        print(f"Generating matplotlib specgram for {detector_label}{' (whitened)' if whitened else ''}...")
        plt.figure(figsize=(10, 5))
        Pxx, freqs, bins, im = plt.specgram(
            strain.value, NFFT=NFFT, Fs=strain.sample_rate.value, noverlap=noverlap,
            cmap='viridis', scale='dB', vmin=-120, vmax=-60
        )
        plt.ylim(fmin, fmax)
        plt.xlabel('Time (s) [relative to start]')
        plt.ylabel('Frequency (Hz)')
        plt.title(f"{detector_label} {'Whitened ' if whitened else ''}Strain Spectrogram (matplotlib)")
        plt.colorbar(label='dB')
        plt.tight_layout()
        fname = f"{detector_label}_{'whitened_' if whitened else ''}specgram_matplotlib.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.show()
        print(f"Matplotlib specgram saved to {os.path.join(output_dir, fname)}")
    except Exception as e:
        print(f"Error generating matplotlib specgram for {detector_label}: {e}")

def plot_gwpy_spectrogram(strain, detector_label, whitened=False):
    try:
        print(f"Generating GWpy spectrogram for {detector_label}{' (whitened)' if whitened else ''}...")
        spec = strain.spectrogram(1)  # 1-second FFTs
        spec = spec.crop(fmin, fmax, frequency=True)
        plot = spec.plot(norm='log', vmin=1e-23, vmax=1e-20, cmap='viridis')
        ax = plot.gca()
        ax.set_title(f"{detector_label} {'Whitened ' if whitened else ''}Strain Spectrogram (GWpy)")
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s) [GPS]')
        plot.colorbar(label='Amplitude [strain/√Hz]')
        plot.tight_layout()
        fname = f"{detector_label}_{'whitened_' if whitened else ''}spectrogram_gwpy.png"
        plot.savefig(os.path.join(output_dir, fname))
        plot.show()
        print(f"GWpy spectrogram saved to {os.path.join(output_dir, fname)}")
    except Exception as e:
        print(f"Error generating GWpy spectrogram for {detector_label}: {e}")

def whiten_strain(strain):
    try:
        print("Whitening strain data...")
        return strain.whiten()
    except Exception as e:
        print(f"Error whitening strain: {e}")
        return None

for strain_proc, label in zip([strain_H1_proc, strain_L1_proc], ['H1', 'L1']):
    # Matplotlib specgram (raw)
    plot_matplotlib_specgram(strain_proc, label, whitened=False)
    # GWpy spectrogram (raw)
    plot_gwpy_spectrogram(strain_proc, label, whitened=False)
    # Whitening
    whitened = whiten_strain(strain_proc)
    if whitened is not None:
        # Matplotlib specgram (whitened)
        plot_matplotlib_specgram(whitened, label, whitened=True)
        # GWpy spectrogram (whitened)
        plot_gwpy_spectrogram(whitened, label, whitened=True)

print("="*60)
print("Analysis complete. All outputs saved in:", output_dir)