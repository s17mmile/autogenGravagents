from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# Center time of GW150914 (GPS time)
event_time = 1126259462.4

# Download ~4 seconds of strain data around GW150914 for Hanford (H1)
data = TimeSeries.fetch_open_data(
    "H1",
    event_time - 2,
    event_time + 2,
    format="hdf5",
)

# Plot
plot = data.plot(
    title="LIGO‑Hanford strain data around GW150914",
    ylabel="Strain $h(t)$",
    xlabel="Time since event (s)",
)
ax = plot.gca()
ax.set_xlim(event_time - 0.2, event_time + 0.2)
ax.axvline(event_time, color="orange", linestyle="--", label="GW150914 time")
ax.legend()
plt.show()