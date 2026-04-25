# filename: gw150914_analysis_final_v10.py
import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from pycbc.waveform import get_td_waveform
from pycbc import filter as pycbc_filter

# Task 1: Data fetching
# Define the event and time parameters
event_name = 'GW150914'
start_offset = -8  # seconds before merger
end_offset = 4     # seconds after merger

# Check if data files exist
h1_file = '../gwosc_gw150914_h1.hdf5'
l1_file = '../gwosc_gw150914_l1.hdf5'
if os.path.exists(h1_file) and os.path.exists(l1_file):
    print('Loading data from disk...')
    h1_data = TimeSeries.read(h1_file)
    l1_data = TimeSeries.read(l1_file)
else:
    print('Fetching data from GWOSC...')
    try:
        gps_time = event_gps(event_name)
    except Exception as e:
        print(f'Error fetching event time: {e}')
        raise
    gps_start = gps_time + start_offset
    gps_end = gps_time + end_offset
    h1_data = TimeSeries.fetch_open_data('H1', gps_start, gps_end)
    l1_data = TimeSeries.fetch_open_data('L1', gps_start, gps_end)
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(h1_file), exist_ok=True)
    # Save to disk
    h1_data.write(h1_file)
    l1_data.write(l1_file)

# Plot strain data
plt.figure(figsize=(10, 5))
plt.plot(h1_data.times, h1_data, label='H1 Strain')
plt.plot(l1_data.times, l1_data, label='L1 Strain')
plt.title('Strain Data for GW150914')
plt.xlabel('Time (GPS)')
plt.ylabel('Strain')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('strain_data.png')
plt.close()

# Task 2: Data filtering
# Whiten the signals
try:
    h1_whitened = h1_data.whiten()
    l1_whitened = l1_data.whiten()
except Exception as e:
    print(f'Error whitening data: {e}')
    raise

# Plot whitened data
plt.figure(figsize=(10, 5))
plt.plot(h1_whitened.times, h1_whitened, label='H1 Whitened')
plt.plot(l1_whitened.times, l1_whitened, label='L1 Whitened')
plt.title('Whitened Strain Data')
plt.xlabel('Time (GPS)')
plt.ylabel('Whitened Strain')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('whitened_strain_data.png')
plt.close()

# Apply band-pass filter
try:
    h1_filtered = h1_whitened.bandpass(30, 250)
    l1_filtered = l1_whitened.bandpass(30, 250)
except Exception as e:
    print(f'Error applying band-pass filter: {e}')
    raise

# Plot filtered data
plt.figure(figsize=(10, 5))
plt.plot(h1_filtered.times, h1_filtered, label='H1 Filtered')
plt.plot(l1_filtered.times, l1_filtered, label='L1 Filtered')
plt.title('Filtered Strain Data')
plt.xlabel('Time (GPS)')
plt.ylabel('Filtered Strain')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('filtered_strain_data.png')
plt.close()

# Task 3: Q-Transform
h1_q_transform = h1_filtered.q_transform()
l1_q_transform = l1_filtered.q_transform()

# Ensure the Q-transform outputs are 2D arrays
if h1_q_transform.ndim != 2 or l1_q_transform.ndim != 2:
    raise ValueError('Q-transform output is not a 2D array.')

plt.figure(figsize=(10, 5))
plt.imshow(h1_q_transform.value, aspect='auto', origin='lower', extent=[h1_q_transform.times[0].value, h1_q_transform.times[-1].value, 0, 25], cmap='inferno')
plt.colorbar(label='Normalized Energy')
plt.title('Q-Transform for H1')
plt.xlabel('Time (GPS)')
plt.ylabel('Frequency (Hz)')
plt.ylim(0, 25)
plt.tight_layout()
plt.savefig('q_transform_h1.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.imshow(l1_q_transform.value, aspect='auto', origin='lower', extent=[l1_q_transform.times[0].value, l1_q_transform.times[-1].value, 0, 25], cmap='inferno')
plt.colorbar(label='Normalized Energy')
plt.title('Q-Transform for L1')
plt.xlabel('Time (GPS)')
plt.ylabel('Frequency (Hz)')
plt.ylim(0, 25)
plt.tight_layout()
plt.savefig('q_transform_l1.png')
plt.close()

# Task 4: Data format conversion
# Ensure consistent sample rates and lengths
h1_resampled = h1_filtered
l1_resampled = l1_filtered

# Task 5: PyCBC template creation
masses = [10, 20, 30, 40]
best_fit_h1 = (None, 0)  # (template, max SNR)
best_fit_l1 = (None, 0)
for mass in masses:
    try:
        template = get_td_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, spin1z=0, spin2z=0, f_lower=20, delta_t=1.0/4096)
        if len(template) < 0.2:
            continue  # Skip templates shorter than 0.2s
        # Scale and truncate/pad
        template = template.crop(0, 0.2)
        plt.figure(figsize=(10, 5))
        plt.plot(h1_resampled.times, h1_resampled, label='H1 Strain')
        plt.plot(template.sample_times, template, label='Template')
        plt.title(f'Template for {mass} Solar Masses')
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'template_{mass}.png')
        plt.close()
        # Matched filtering
        h1_mf = pycbc_filter.match(h1_resampled, template)
        l1_mf = pycbc_filter.match(l1_resampled, template)
        # Track best fit
        max_h1 = np.max(h1_mf)
        max_l1 = np.max(l1_mf)
        if max_h1 > best_fit_h1[1]:
            best_fit_h1 = (template, max_h1)
        if max_l1 > best_fit_l1[1]:
            best_fit_l1 = (template, max_l1)
    except Exception as e:
        print(f'Error generating template for mass {mass}: {e}')
        continue

print(f'Best fit template for H1: {best_fit_h1[0]} with SNR: {best_fit_h1[1]}')
print(f'Best fit template for L1: {best_fit_l1[0]} with SNR: {best_fit_l1[1]}')

# Task 6: Calculating the Power Spectral Density
try:
    h1_psd = h1_filtered.psd(average='mean')
    l1_psd = l1_filtered.psd(average='mean')
except Exception as e:
    print(f'Error calculating PSD: {e}')
    raise

# Plot PSDs
plt.figure(figsize=(10, 5))
plt.loglog(h1_psd.frequencies, h1_psd.value, label='H1 PSD')
plt.loglog(l1_psd.frequencies, l1_psd.value, label='L1 PSD')
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('psd.png')
plt.close()

# Task 7: Matched filtering
for mass in masses:
    try:
        template = get_td_waveform(approximant='SEOBNRv4_opt', mass1=mass, mass2=mass, spin1z=0, spin2z=0, f_lower=20, delta_t=1.0/4096)
        h1_mf = pycbc_filter.match(h1_resampled, template)
        l1_mf = pycbc_filter.match(l1_resampled, template)
        plt.figure(figsize=(10, 5))
        plt.plot(h1_mf.sample_times, h1_mf, label='H1 Matched Filter')
        plt.title(f'Matched Filtering for {mass} Solar Masses (H1)')
        plt.xlabel('Time (s)')
        plt.ylabel('SNR')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'matched_filter_h1_{mass}.png')
        plt.close()
        plt.figure(figsize=(10, 5))
        plt.plot(l1_mf.sample_times, l1_mf, label='L1 Matched Filter')
        plt.title(f'Matched Filtering for {mass} Solar Masses (L1)')
        plt.xlabel('Time (s)')
        plt.ylabel('SNR')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'matched_filter_l1_{mass}.png')
        plt.close()
    except Exception as e:
        print(f'Error during matched filtering for mass {mass}: {e}')