from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 1. Load the GWF strain data
# ------------------------------
fname = "/home/sr/Downloads/H-H1_GWOSC_16KHZ_R1-1186741846-32.gwf"
channel = "H1:GWOSC-16KHZ_R1_STRAIN"
strain = TimeSeries.read(fname, channel)

# Extract delta_t (sampling interval)
delta_t = strain.dt.value  # in seconds
sample_rate = strain.sample_rate.value  # in Hz
print(f"delta_t = {delta_t:.6e} s  (sampling rate = {sample_rate} Hz)")

# ------------------------------
# 2. Bandpass + whiten
# ------------------------------
strain_bp = strain.bandpass(30, 500).whiten()

# ------------------------------
# 3. Q-transform (time-frequency spectrogram)
# ------------------------------
qgram = strain_bp.q_transform(frange=(20, 512))

# ------------------------------
# 4. Plot strain vs time + Q-transform
# ------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ---- Left: Strain vs time ----
scale_factor = 1e21  # scale to "micro-strain"
ax1.plot(strain_bp.times.value, strain_bp.value * scale_factor, color="black", lw=0.5)
ax1.set_title(f"Whitened Strain vs Time (H1)\nΔt = {delta_t:.6e} s, fs = {sample_rate} Hz")
ax1.set_xlabel(f"Time [s] from GPS {strain_bp.epoch.gps}")
ax1.set_ylabel("Strain × 10^-21")

# Mark midpoint of full data
center_time = float(strain_bp.times.value[0] + strain_bp.duration.value / 2)
ax1.axvline(center_time, color='red', ls='--', lw=2, label='File center')
ax1.legend()

# ---- Right: Q-transform spectrogram ----
t = qgram.times.value
f = qgram.frequencies.value
im = ax2.pcolormesh(t, f, qgram.value.T, shading='auto', vmin=0, vmax=25)

ax2.set_title("Q-transform Spectrogram (H1 strain)")
ax2.set_xlabel(f"Time [s] from GPS {strain_bp.epoch.gps}")
ax2.set_ylabel("Frequency [Hz]")

# Vertical line on Q-transform
ax2.axvline(center_time, color='red', ls='--', lw=2)

# Colorbar
cbar = fig.colorbar(im, ax=ax2, label="Normalized energy")

plt.tight_layout()
plt.show()
