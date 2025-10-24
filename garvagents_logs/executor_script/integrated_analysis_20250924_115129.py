# ============================================================
# Gravitational Wave Event Strain Analysis and Visualization
# ============================================================

# --------------- Imports and Setup ---------------
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
from pycbc.catalog import Merger
from pycbc.frame import query_and_read_frame
from pycbc.types import TimeSeries

# --------------- Parameters and Event List ---------------
EVENT_NAMES = [
    "GW150914",
    "GW151226",
    "GW170104",
    "GW170608",
    "GW170814",
    "GW190521"
]
WINDOW = 8  # seconds of data around event time
HALF_WINDOW = WINDOW // 2

# Bandpass filter parameters
LOWCUT = 35.0   # Hz
HIGHCUT = 350.0 # Hz
ORDER = 4       # Butterworth filter order

# Spectrogram parameters
NPERSEG = 256
NOVERLAP = 128

# Output directories
RAW_DATA_DIR = "strain_data_raw"
FILTERED_DATA_DIR = "strain_data_filtered"
PLOT_DIR = "strain_plots"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(FILTERED_DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# --------------- Task 1: Download and Load Strain Data ---------------
print("\n========== TASK 1: Downloading and Loading Strain Data ==========")
strain_data = {}

for event_name in EVENT_NAMES:
    try:
        print(f"\nProcessing event: {event_name}")
        merger = Merger(event_name)
        event_time = merger.time
        # FIX: Use 'instruments' instead of 'detectors'
        detectors = getattr(merger, 'instruments', None)
        if detectors is None:
            print(f"  Warning: No detectors/instruments found for {event_name}. Skipping.")
            strain_data[event_name] = None
            continue
        print(f"  Event time: {event_time}")
        print(f"  Detectors: {detectors}")

        strain_data[event_name] = {}

        for det in detectors:
            print(f"    Downloading strain for detector: {det}")
            try:
                ts = query_and_read_frame(
                    observatory=det,
                    channel=f"{det}:GWOSC-4KHZ_R1_STRAIN",
                    start_time=event_time - HALF_WINDOW,
                    end_time=event_time + HALF_WINDOW
                )
                strain_data[event_name][det] = ts
                print(f"      Loaded strain data: {len(ts)} samples, dt={ts.delta_t}")

                # Save raw strain data to disk for reproducibility
                np.savez_compressed(
                    os.path.join(RAW_DATA_DIR, f"{event_name}_{det}_raw.npz"),
                    strain=ts.numpy(),
                    times=ts.sample_times.numpy(),
                    delta_t=ts.delta_t
                )
            except Exception as e:
                print(f"      Error loading strain for {det}: {e}")
                strain_data[event_name][det] = None

    except Exception as e:
        print(f"  Error processing event {event_name}: {e}")
        strain_data[event_name] = None

print("\nAll requested strain data loaded (where available).")

# --------------- Task 2: Bandpass Filter the Strain Data ---------------
print("\n========== TASK 2: Bandpass Filtering Strain Data ==========")
filtered_strain_data = {}

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

for event_name, detectors in strain_data.items():
    filtered_strain_data[event_name] = {}
    if detectors is None:
        print(f"Skipping event {event_name} (no data loaded).")
        continue
    for det, ts in detectors.items():
        if ts is None:
            print(f"  Skipping {event_name} {det} (no strain data).")
            filtered_strain_data[event_name][det] = None
            continue
        try:
            print(f"Filtering {event_name} {det}...")
            fs = 1.0 / ts.delta_t
            b, a = butter_bandpass(LOWCUT, HIGHCUT, fs, order=ORDER)
            filtered = filtfilt(b, a, ts.numpy())
            filtered_ts = ts.copy()
            filtered_ts.data = np.array(filtered, dtype=filtered_ts.data.dtype)
            filtered_strain_data[event_name][det] = filtered_ts
            print(f"  Filtered {event_name} {det}: {len(filtered_ts)} samples.")

            # Save filtered strain data to disk
            np.savez_compressed(
                os.path.join(FILTERED_DATA_DIR, f"{event_name}_{det}_filtered.npz"),
                strain=filtered_ts.numpy(),
                times=filtered_ts.sample_times.numpy(),
                delta_t=filtered_ts.delta_t
            )
        except Exception as e:
            print(f"  Error filtering {event_name} {det}: {e}")
            filtered_strain_data[event_name][det] = None

print("\nBandpass filtering complete for all available strain data.")

# --------------- Task 3: Plot Strain and Spectrograms ---------------
print("\n========== TASK 3: Plotting Strain and Spectrograms ==========")
for event_name, detectors in filtered_strain_data.items():
    if detectors is None:
        print(f"Skipping event {event_name} (no filtered data).")
        continue
    for det, ts in detectors.items():
        if ts is None:
            print(f"  Skipping {event_name} {det} (no filtered strain data).")
            continue
        try:
            print(f"Plotting {event_name} {det}...")
            times = ts.sample_times.numpy()
            strain = ts.numpy()
            fs = 1.0 / ts.delta_t

            fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
            fig.suptitle(f"{event_name} - {det}")

            # Time-domain strain
            axs[0].plot(times, strain, color='black', lw=0.7)
            axs[0].set_ylabel("Strain")
            axs[0].set_title("Filtered Strain Time Series")
            axs[0].grid(True)

            # Spectrogram
            f, t, Sxx = spectrogram(strain, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)
            im = axs[1].pcolormesh(times[0] + t, f, np.log10(Sxx + 1e-20), shading='auto', cmap='viridis')
            axs[1].set_ylabel("Frequency [Hz]")
            axs[1].set_xlabel("Time [s]")
            axs[1].set_title("Spectrogram (STFT)")
            axs[1].set_ylim(20, 400)
            fig.colorbar(im, ax=axs[1], label='log10(Power)')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save plot to disk
            plot_filename = os.path.join(PLOT_DIR, f"{event_name}_{det}_strain_spectrogram.png")
            plt.savefig(plot_filename)
            print(f"  Saved plot to {plot_filename}")

            plt.show()
        except Exception as e:
            print(f"  Error plotting {event_name} {det}: {e}")

print("\nPlotting complete for all available filtered strain data.")

# --------------- End of Script ---------------