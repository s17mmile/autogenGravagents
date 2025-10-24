# ============================================================
# GW170608 Strain Data Download, Time Series Plot, and Q-transform Spectrogram
# ============================================================

# ------------------ Imports ------------------
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np

# ------------------ Parameters ------------------
# GW170608 GPS time and interval
GPS_EVENT = 1180922494.5  # GW170608 event GPS time
INTERVAL = 32             # seconds before and after

gps_start = GPS_EVENT - INTERVAL
gps_end = GPS_EVENT + INTERVAL

detectors = ['H1', 'L1']

# ------------------ Task 1: Data Loading ------------------
print("="*60)
print("Task 1: Downloading strain data for GW170608 (±32s interval)")
print("="*60)

strain_data = {}

for det in detectors:
    print(f"Attempting to download strain data for {det} from {gps_start} to {gps_end}...")
    try:
        ts = TimeSeries.fetch_open_data(det, gps_start, gps_end, cache=True)
        strain_data[det] = ts
        print(f"  ✔ Successfully downloaded strain data for {det}.")
    except Exception as e:
        print(f"  ✖ Error downloading data for {det}: {e}")
        strain_data[det] = None

if not any(strain_data[det] is not None for det in detectors):
    raise RuntimeError("No strain data could be downloaded for any detector. Exiting.")

# ------------------ Task 2: Strain Time Series Plot ------------------
print("\n" + "="*60)
print("Task 2: Plotting strain time series")
print("="*60)

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plot_success = False

for idx, det in enumerate(detectors):
    ts = strain_data.get(det)
    if ts is not None:
        print(f"Plotting strain data for {det}...")
        try:
            # Plot time relative to GPS_EVENT
            axs[idx].plot(ts.times.value - GPS_EVENT, ts.value, label=f"{det} strain", color=f"C{idx}")
            axs[idx].set_ylabel('Strain')
            axs[idx].legend(loc='upper right')
            axs[idx].set_title(f"{det} Strain Data")
            plot_success = True
        except Exception as e:
            print(f"  ✖ Error plotting data for {det}: {e}")
    else:
        print(f"  ✖ No strain data available for {det}, skipping plot.")

axs[-1].set_xlabel('Time (s) relative to GW170608 GPS ({:.1f})'.format(GPS_EVENT))
plt.suptitle("GW170608 Strain Data (±32s interval)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

if plot_success:
    plt.show()
    try:
        fig.savefig("GW170608_strain_timeseries.png")
        print("  ✔ Strain time series plot saved as 'GW170608_strain_timeseries.png'.")
    except Exception as e:
        print(f"  ✖ Error saving strain time series plot: {e}")
else:
    print("  ✖ No plots were generated due to missing data.")

# ------------------ Task 3: Q-transform Spectrogram ------------------
print("\n" + "="*60)
print("Task 3: Computing and plotting Q-transform spectrograms")
print("="*60)

qtransforms = {}

for det in detectors:
    ts = strain_data.get(det)
    if ts is not None:
        print(f"Computing Q-transform for {det}...")
        try:
            # Compute Q-transform (default parameters are suitable for event visualization)
            q = ts.q_transform()
            qtransforms[det] = q
            print(f"  ✔ Q-transform computed for {det}. Plotting spectrogram...")
            # Plot and customize
            fig = q.plot(figsize=(12, 6))
            ax = fig.gca()
            ax.set_title(f"{det} Q-transform Spectrogram (GW170608, ±32s)")
            ax.set_ylabel('Frequency [Hz]')
            ax.set_xlabel('Time [s] relative to GW170608 GPS ({:.1f})'.format(GPS_EVENT))
            # Set x-axis to be relative to GPS_EVENT
            xlim = ax.get_xlim()
            ax.set_xlim(xlim[0] - GPS_EVENT, xlim[1] - GPS_EVENT)
            plt.tight_layout()
            plt.show()
            # Save the figure
            filename = f"GW170608_{det}_qtransform.png"
            try:
                fig.savefig(filename)
                print(f"  ✔ Spectrogram saved as '{filename}'.")
            except Exception as e:
                print(f"  ✖ Error saving spectrogram for {det}: {e}")
        except Exception as e:
            print(f"  ✖ Error computing or plotting Q-transform for {det}: {e}")
            qtransforms[det] = None
    else:
        print(f"  ✖ No strain data available for {det}, skipping Q-transform.")

print("\nAll tasks completed.")