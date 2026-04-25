# filename: gw150914_analysis_improved_fixed_v2.py

import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc import datasets
from pycbc import waveform, psd, filter as pycbc_filter
from pycbc import types as pycbc_types


def fetch_data(event_name, start_offset, end_offset):
    """Fetches strain data for the specified event from GWOSC or loads from disk if available."""
    h1_file = '../gwosc_gw150914_h1.hdf5'
    l1_file = '../gwosc_gw150914_l1.hdf5'

    if os.path.exists(h1_file) and os.path.exists(l1_file):
        print('Loading data from disk...')
        h1_data = TimeSeries.read(h1_file)
        l1_data = TimeSeries.read(l1_file)
    else:
        try:
            print('Fetching event time from GWOSC...')
            gps_time = datasets.get_event_time(event_name)
            start_time = gps_time + start_offset
            end_time = gps_time + end_offset
            print(f'Fetching data from {start_time} to {end_time}...')
            h1_data = TimeSeries.fetch_open_data('H1', start_time, end_time)
            l1_data = TimeSeries.fetch_open_data('L1', start_time, end_time)
            os.makedirs(os.path.dirname(h1_file), exist_ok=True)
            h1_data.write(h1_file)
            l1_data.write(l1_file)
        except Exception as e:
            print(f'Error fetching data: {e}')
            return None, None
    return h1_data, l1_data


def plot_strain(h1_data, l1_data):
    """Plots the strain data for both detectors and saves the figure."""
    plt.figure(figsize=(12, 6))
    plt.plot(h1_data.times, h1_data, label='H1 Strain')
    plt.plot(l1_data.times, l1_data, label='L1 Strain')
    plt.title('Strain Data for GW150914')
    plt.xlabel('GPS Time (s)')
    plt.ylabel('Strain')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('strain_data.png')
    plt.close()


def filter_data(h1_data, l1_data):
    """Whiten and band-pass filter the strain data."""
    h1_whitened = h1_data.whiten()
    l1_whitened = l1_data.whiten()

    h1_filtered = h1_whitened.bandpass(30, 250)
    l1_filtered = l1_whitened.bandpass(30, 250)

    return h1_filtered, l1_filtered


def plot_filtered_data(h1_filtered, l1_filtered):
    """Plots the filtered strain data and saves the figure."""
    plt.figure(figsize=(12, 6))
    plt.plot(h1_filtered.times, h1_filtered, label='H1 Filtered')
    plt.plot(l1_filtered.times, l1_filtered, label='L1 Filtered')
    plt.title('Filtered Strain Data')
    plt.xlabel('GPS Time (s)')
    plt.ylabel('Filtered Strain')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('filtered_strain_data.png')
    plt.close()


def q_transform(h1_filtered, l1_filtered):
    """Creates Q-transform plots for both detectors and saves them."""
    plt.figure(figsize=(12, 6))
    plt.specgram(h1_filtered, NFFT=1024, Fs=1.0, noverlap=512, cmap='inferno')
    plt.colorbar(label='Energy')
    plt.title('Q-Transform for H1')
    plt.ylim(0, 25)
    plt.tight_layout()
    plt.savefig('q_transform_h1.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.specgram(l1_filtered, NFFT=1024, Fs=1.0, noverlap=512, cmap='inferno')
    plt.colorbar(label='Energy')
    plt.title('Q-Transform for L1')
    plt.ylim(0, 25)
    plt.tight_layout()
    plt.savefig('q_transform_l1.png')
    plt.close()


def convert_to_pycbc(h1_filtered, l1_filtered):
    """Converts filtered data to PyCBC TimeSeries format and ensures consistent sample rates."""
    delta_t_h1 = 1 / h1_filtered.sample_rate
    delta_t_l1 = 1 / l1_filtered.sample_rate

    h1_pycbc = pycbc_types.TimeSeries(h1_filtered, delta_t=delta_t_h1)
    l1_pycbc = pycbc_types.TimeSeries(l1_filtered, delta_t=delta_t_l1)

    if h1_pycbc.sample_rate != l1_pycbc.sample_rate:
        print('Resampling L1 to match H1 sample rate...')
        l1_pycbc = l1_pycbc.resample(h1_pycbc.sample_rate)

    return h1_pycbc, l1_pycbc


def create_templates(masses, approximant):
    """Generates waveform templates for specified masses and returns them."""
    templates = []
    for mass in masses:
        template = waveform.get_waveform(approximant=approximant, mass1=mass, mass2=mass, spin1z=0, spin2z=0)
        if template is not None:
            templates.append(template)
    return templates


def plot_templates(templates, h1_pycbc):
    """Plots each template over the strain data and saves the figures."""
    for template in templates:
        if len(template) > 0.2 * h1_pycbc.duration:
            template = template.crop(0, h1_pycbc.duration)
            template *= (h1_pycbc.max() / template.max())
            plt.figure(figsize=(12, 6))
            plt.plot(h1_pycbc.times, h1_pycbc, label='H1 Strain')
            plt.plot(template.times, template, label='Template', alpha=0.7)
            plt.title(f'Template for Mass {mass} Solar Masses')
            plt.xlabel('GPS Time (s)')
            plt.ylabel('Strain')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f'template_mass_{mass}.png')
            plt.close()


def calculate_psd(h1_pycbc, l1_pycbc):
    """Calculates and plots the Power Spectral Density for both detectors."""
    h1_psd = h1_pycbc.psd(4)  # 4 seconds segment length
    l1_psd = l1_pycbc.psd(4)

    plt.figure(figsize=(12, 6))
    plt.loglog(h1_psd.sample_frequencies, h1_psd, label='H1 PSD')
    plt.loglog(l1_psd.sample_frequencies, l1_psd, label='L1 PSD')
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('psd.png')
    plt.close()


def matched_filtering(templates, h1_pycbc, l1_pycbc):
    """Performs matched filtering and plots SNR for each template and detector."""
    best_fit = {'H1': None, 'L1': None}
    for template in templates:
        h1_snr = pycbc_filter.match(h1_pycbc, template)
        l1_snr = pycbc_filter.match(l1_pycbc, template)

        plt.figure(figsize=(12, 6))
        plt.plot(h1_snr.sample_times, h1_snr, label='H1 SNR')
        plt.title(f'SNR for Template Mass {mass} (H1)')
        plt.xlabel('GPS Time (s)')
        plt.ylabel('SNR')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'snr_h1_mass_{mass}.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(l1_snr.sample_times, l1_snr, label='L1 SNR')
        plt.title(f'SNR for Template Mass {mass} (L1)')
        plt.xlabel('GPS Time (s)')
        plt.ylabel('SNR')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'snr_l1_mass_{mass}.png')
        plt.close()

        # Update best fit
        if best_fit['H1'] is None or h1_snr.max() > best_fit['H1'][1]:
            best_fit['H1'] = (mass, h1_snr.max())
        if best_fit['L1'] is None or l1_snr.max() > best_fit['L1'][1]:
            best_fit['L1'] = (mass, l1_snr.max())

    print(f"Best fit for H1: Mass {best_fit['H1'][0]} with SNR {best_fit['H1'][1]}")
    print(f"Best fit for L1: Mass {best_fit['L1'][0]} with SNR {best_fit['L1'][1]}")


def main():
    event_name = 'GW150914'
    start_offset = -8  # seconds before merger
    end_offset = 4     # seconds after merger

    h1_data, l1_data = fetch_data(event_name, start_offset, end_offset)
    if h1_data is None or l1_data is None:
        return

    plot_strain(h1_data, l1_data)
    h1_filtered, l1_filtered = filter_data(h1_data, l1_data)
    plot_filtered_data(h1_filtered, l1_filtered)
    q_transform(h1_filtered, l1_filtered)

    h1_pycbc, l1_pycbc = convert_to_pycbc(h1_filtered, l1_filtered)
    masses = [10, 20, 30, 40]
    approximant = 'SEOBNRv4_opt'
    templates = create_templates(masses, approximant)
    plot_templates(templates, h1_pycbc)
    calculate_psd(h1_pycbc, l1_pycbc)
    matched_filtering(templates, h1_pycbc, l1_pycbc)

if __name__ == '__main__':
    main()