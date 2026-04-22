# filename: gw150914_analysis_revised_fixed_v10.py
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.signal.qtransform import QGram
from pycbc import waveform
from gwosc.datasets import event_gps
import os

def plot_data(data, title, filename):
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel('GPS Time (s)')
    plt.ylabel('Strain')
    plt.savefig(filename)
    plt.close()

# Task 1: Data fetching
try:
    # Get the GPS time for the GW150914 event
    event_time = event_gps('GW150914')
    start_time = event_time - 8  # 8 seconds before the event
    end_time = event_time + 4    # 4 seconds after the event
    
    # Fetch L1 and H1 strain data
    l1_data = TimeSeries.fetch_open_data('L1', start_time, end_time)
    h1_data = TimeSeries.fetch_open_data('H1', start_time, end_time)
except Exception as e:
    print(f'Error fetching data: {e}')
    raise

# Task 2: Data filtering
# Whiten the data
l1_whitened = l1_data.whiten()
h1_whitened = h1_data.whiten()

# Apply band-pass filter
l1_filtered = l1_whitened.bandpass(30, 250)
h1_filtered = h1_whitened.bandpass(30, 250)

# Save filtered data plots
plot_data(l1_filtered, 'L1 Filtered Data', 'l1_filtered_data.png')
plot_data(h1_filtered, 'H1 Filtered Data', 'h1_filtered_data.png')

# Task 3: Q-Transform
# Create Q-transform plots
plt.figure()
QGram(l1_filtered, dt=1/l1_filtered.sample_rate, fmin=30, fmax=250)
plt.colorbar(label='Normalized Energy')
plt.title('L1 Q-Transform')
plt.xlabel('GPS Time (s)')
plt.ylabel('Frequency (Hz)')
plt.savefig('l1_q_transform.png')
plt.close()

plt.figure()
QGram(h1_filtered, dt=1/h1_filtered.sample_rate, fmin=30, fmax=250)
plt.colorbar(label='Normalized Energy')
plt.title('H1 Q-Transform')
plt.xlabel('GPS Time (s)')
plt.ylabel('Frequency (Hz)')
plt.savefig('h1_q_transform.png')
plt.close()

# Task 4: Template creation
# Generate waveform templates
masses = np.arange(10, 31, 1)
templates = []
for mass1 in masses:
    for mass2 in masses:
        hp, hc = waveform.get_td_waveform(approximant='SEOBNRv4_opt', mass1=mass1, mass2=mass2, delta_t=1/h1_filtered.sample_rate)
        if len(hp) > int(0.2 * h1_filtered.sample_rate):  # Check length
            templates.append(hp)

# Scale templates and plot
for i, template in enumerate(templates):
    scaled_template = template / np.max(np.abs(template)) * np.max(np.abs(h1_filtered))
    plt.figure()
    plt.plot(h1_filtered, label='H1 Filtered Data')
    plt.plot(scaled_template, label='Scaled Template')
    plt.title(f'Template Overlay for Mass {masses[i]}')
    plt.xlabel('GPS Time (s)')
    plt.ylabel('Strain')
    plt.legend()
    plt.savefig(f'template_overlay_mass_{masses[i]}.png')
    plt.close()

# Combined H1 strain plot
plot_data(h1_filtered, 'Combined H1 Strain Data', 'combined_h1_strain.png')