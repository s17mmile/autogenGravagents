import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries
from pycbc.psd import welch
from pycbc.vetoes import power_chisq

# Step 1: Generate a gravitational wave signal
print("Generating gravitational wave signal...")
hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                         mass1=10,
                         mass2=10,
                         delta_t=1.0/4096,
                         f_lower=30)

# Step 2: Create a noisy time series (simulate detector data)
print("Simulating detector noise...")
duration = 8  # seconds
sample_rate = 4096  # Hz
n_samples = duration * sample_rate
np.random.seed(42)
noise = np.random.normal(0, 1, n_samples)

# Construct PyCBC TimeSeries correctly
try:
    data = TimeSeries(noise, delta_t=1.0/sample_rate, epoch=0)
except Exception as e:
    print(f"Error constructing TimeSeries: {e}")
    raise

# Step 3: Estimate the Power Spectral Density (PSD) using Welch's method
print("Estimating PSD using Welch's method...")
# seg_len must be <= half the data duration and >= 2
max_seg_len = max(2, n_samples // 2)
default_seg_len = 4 * sample_rate  # 4 seconds
seg_len = min(default_seg_len, max_seg_len)

# Try to estimate PSD, fallback to smaller seg_len if needed
psd = None
while seg_len >= 2:
    try:
        psd = welch(data, seg_len=seg_len)
        psd.resize(len(data))  # In-place resize
        print(f"PSD estimated with seg_len={seg_len}")
        break
    except ValueError as e:
        print(f"ValueError in PSD estimation with seg_len={seg_len}: {e}")
        seg_len = seg_len // 2  # Try smaller segment
    except Exception as e:
        print(f"Unexpected error in PSD estimation: {e}")
        raise

if psd is None:
    raise RuntimeError("Failed to estimate PSD: no valid seg_len found.")

# Step 4: Compute the chi-squared veto statistic using power_chisq
print("Computing chi-squared veto statistic...")
try:
    # For demonstration, use the signal as the template
    # Truncate or pad the template to match the data length
    template = hp.copy()
    if len(template) > len(data):
        template = template[:len(data)]
    elif len(template) < len(data):
        template = template.copy()
        template.resize(len(data))
    # Compute power_chisq
    chisq, dof = power_chisq(template, data, psd, 16)
    print(f"Chi-squared value: {chisq}, Degrees of freedom: {dof}")
except Exception as e:
    print(f"Error computing chi-squared: {e}")
    raise

# Step 5: Plot the data and template for visualization
plt.figure(figsize=(10, 4))
plt.plot(data.sample_times, data, label="Noisy Data")
plt.plot(data.sample_times, template, label="Template", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.legend()
plt.title("Simulated Detector Data and Template")
plt.tight_layout()
plt.show()