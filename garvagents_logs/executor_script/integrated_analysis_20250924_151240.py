# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

# --- Constants and Parameters ---
# GW170608 event
EVENT_GPS = 1180922494.5
DURATION = 128  # seconds (±64 s)
HALF_DURATION = DURATION // 2
DETECTORS = ['H1', 'L1']

# Preprocessing parameters
BANDPASS_LOW = 35
BANDPASS_HIGH = 350

# Visualization parameters
FULL_WINDOW = 64  # seconds before and after event
ZOOM_WINDOW = 0.5  # seconds before and after event

# Q-transform parameters
Q_FMIN = 20
Q_FMAX = 400
Q_QRANGE = (8, 32)

# Output file templates
FULL_FIG_TEMPLATE = "strain_full_{det}.png"
ZOOM_FIG_TEMPLATE = "strain_zoom_{det}.png"
QTR_FIG_TEMPLATE = "qtransform_{det}.png"

# --- 1. Data Loading ---
print("="*60)
print("STEP 1: Downloading strain data for GW170608 (H1 and L1)")
print("="*60)
strain_data = {}

for det in DETECTORS:
    print(f"\nFetching data for {det}...")
    try:
        ts = TimeSeries.fetch_open_data(
            det,
            EVENT_GPS - HALF_DURATION,
            EVENT_GPS + HALF_DURATION,
            cache=True,
            verbose=True
        )
        print(f"Successfully fetched data for {det}.")
        if ts.has_gaps:
            print(f"Warning: Data for {det} contains gaps. Gaps: {ts.gaps}")
        else:
            print(f"No gaps detected in {det} data.")
        strain_data[det] = ts
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        strain_data[det] = None

print("\nData loading complete.")

# --- 2. Preprocessing ---
print("\n" + "="*60)
print("STEP 2: Preprocessing (bandpass filter and whitening)")
print("="*60)
processed_data = {}

for det, ts in strain_data.items():
    print(f"\nProcessing data for {det}...")
    if ts is None:
        print(f"Skipping {det}: No data available.")
        processed_data[det] = None
        continue
    try:
        print(f"Applying bandpass filter ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz) to {det}...")
        ts_bp = ts.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
        print(f"Bandpass filter applied to {det}.")
        print(f"Whitening the bandpassed data for {det}...")
        ts_whitened = ts_bp.whiten()
        print(f"Whitening complete for {det}.")
        processed_data[det] = ts_whitened
    except Exception as e:
        print(f"Error processing {det}: {e}")
        processed_data[det] = None

print("\nPreprocessing complete.")

# --- 3. Time Series Visualization ---
print("\n" + "="*60)
print("STEP 3: Time Series Visualization")
print("="*60)

for det, ts in processed_data.items():
    print(f"\nPlotting for {det}...")
    if ts is None:
        print(f"Skipping {det}: No processed data available.")
        continue
    try:
        # Convert time axis to seconds relative to event
        time_rel = ts.times.value - EVENT_GPS
        strain = ts.value

        # --- Full ±64 s plot ---
        print(f"Plotting full ±64 s time series for {det}...")
        plt.figure(figsize=(12, 4))
        plt.plot(time_rel, strain, color='C0', lw=0.5)
        plt.title(f"{det} Strain Data (Whitened, Bandpassed)\nGW170608, Full ±64 s")
        plt.xlabel("Time (s) relative to event")
        plt.ylabel("Strain")
        plt.xlim(-FULL_WINDOW, FULL_WINDOW)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        full_fig_name = FULL_FIG_TEMPLATE.format(det=det)
        plt.savefig(full_fig_name, dpi=150)
        plt.show()
        print(f"Saved full time series plot to {full_fig_name}")

        # --- Zoomed ±0.5 s plot ---
        print(f"Plotting zoomed ±0.5 s time series for {det}...")
        zoom_mask = (time_rel >= -ZOOM_WINDOW) & (time_rel <= ZOOM_WINDOW)
        plt.figure(figsize=(12, 4))
        plt.plot(time_rel[zoom_mask], strain[zoom_mask], color='C1', lw=1)
        plt.title(f"{det} Strain Data (Whitened, Bandpassed)\nGW170608, Zoomed ±0.5 s")
        plt.xlabel("Time (s) relative to event")
        plt.ylabel("Strain")
        plt.xlim(-ZOOM_WINDOW, ZOOM_WINDOW)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        zoom_fig_name = ZOOM_FIG_TEMPLATE.format(det=det)
        plt.savefig(zoom_fig_name, dpi=150)
        plt.show()
        print(f"Saved zoomed time series plot to {zoom_fig_name}")

    except Exception as e:
        print(f"Error plotting for {det}: {e}")

print("\nTime series visualization complete.")

# --- 4. Q-transform Visualization ---
print("\n" + "="*60)
print("STEP 4: Q-transform Spectrogram Visualization")
print("="*60)

for det, ts in processed_data.items():
    print(f"\nComputing Q-transform for {det}...")
    if ts is None:
        print(f"Skipping {det}: No processed data available.")
        continue
    try:
        print(f"Running Q-transform (f={Q_FMIN}-{Q_FMAX} Hz, Q={Q_QRANGE[0]}–{Q_QRANGE[1]}) for {det}...")
        qspec = ts.q_transform(frange=(Q_FMIN, Q_FMAX), qrange=Q_QRANGE)
        print(f"Q-transform complete for {det}.")

        print(f"Plotting Q-transform spectrogram for {det}...")
        fig = qspec.plot(figsize=(12, 5), vmin=0, vmax=15)
        ax = fig.gca()
        ax.set_title(f"{det} Q-transform Spectrogram\nGW170608, {Q_FMIN}-{Q_FMAX} Hz, Q={Q_QRANGE[0]}–{Q_QRANGE[1]}")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s] relative to event")
        # Set time axis relative to event
        ax.set_xlim(EVENT_GPS - FULL_WINDOW, EVENT_GPS + FULL_WINDOW)
        # Convert x-axis to relative time
        ticks = ax.get_xticks()
        ax.set_xticklabels([f"{tick - EVENT_GPS:.1f}" for tick in ticks])
        plt.tight_layout()
        qtr_fig_name = QTR_FIG_TEMPLATE.format(det=det)
        plt.savefig(qtr_fig_name, dpi=150)
        plt.show()
        print(f"Saved Q-transform spectrogram to {qtr_fig_name}")

    except Exception as e:
        print(f"Error computing or plotting Q-transform for {det}: {e}")

print("\nQ-transform visualization complete.")
print("\nAll steps completed successfully.")