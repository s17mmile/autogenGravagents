# filename: automobile_oscillation_amplitude_debug.py
import math

# Given parameters
mass = 1000.0  # kg
settling_per_100kg = 0.01  # m (1.0 cm)
g = 9.8  # m/s^2
speed_kmh = 20.0  # km/h
speed_ms = speed_kmh * 1000 / 3600  # m/s
road_amplitude = 0.05  # m (5.0 cm)
road_wavelength = 0.20  # m (20 cm)
wheelbase = 2.4  # m (distance between front and rear wheels)

# Calculate spring constant k from static deflection per 100 kg
mass_per_100kg = 100.0  # kg
k = (mass_per_100kg * g) / settling_per_100kg  # N/m

# Natural angular frequency of the car's vertical oscillation
omega_0 = math.sqrt(k / mass)  # rad/s

# Driving angular frequency from road profile and car speed
omega_drive = 2 * math.pi * speed_ms / road_wavelength  # rad/s

# Calculate the argument for sine term
arg = omega_drive * wheelbase / (2 * speed_ms)

# Calculate sine of the argument
sin_arg = math.sin(arg)

# Driving displacement amplitude (difference between front and rear wheels)
delta_y_max = 2 * road_amplitude * abs(sin_arg)

# Calculate amplitude of oscillation using undamped driven harmonic oscillator formula
amplitude = delta_y_max / abs(1 - (omega_drive / omega_0)**2)  # m

# Convert amplitude to mm for readability
amplitude_mm = amplitude * 1000

# Output results with debug info
print(f"Spring constant k: {k:.2f} N/m")
print(f"Natural frequency omega_0: {omega_0:.2f} rad/s")
print(f"Driving frequency omega_drive: {omega_drive:.2f} rad/s")
print(f"Sine argument (radians): {arg:.4f}")
print(f"Sine of argument: {sin_arg:.6f}")
print(f"Driving displacement amplitude delta_y_max: {delta_y_max:.8f} m")
print(f"Amplitude of oscillation: {amplitude:.8f} m ({amplitude_mm:.4f} mm)")
