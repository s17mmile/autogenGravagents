# filename: vibrational_population_I2.py
import numpy as np
from scipy.constants import h, c, k

def vibrational_energy(n, freq_cm):
    # Convert frequency from cm^-1 to Hz
    # 1 cm^-1 = c * 100 Hz since c is in m/s and 1 cm = 0.01 m
    freq_hz = freq_cm * c * 100
    # Energy of vibrational level n
    return h * freq_hz * (n + 0.5)

def partition_function(freq_cm, T, max_n=100):
    energies = np.array([vibrational_energy(n, freq_cm) for n in range(max_n)])
    boltzmann_factors = np.exp(-energies / (k * T))
    return np.sum(boltzmann_factors)

def vibrational_level_probability(n, freq_cm, T):
    E_n = vibrational_energy(n, freq_cm)
    Z = partition_function(freq_cm, T)
    P_n = np.exp(-E_n / (k * T)) / Z
    return P_n

if __name__ == "__main__":
    freq_cm = 208  # vibrational frequency in cm^-1
    T = 298  # temperature in K
    n = 2  # vibrational level

    probability_n2 = vibrational_level_probability(n, freq_cm, T)

    try:
        with open('vibrational_probability.txt', 'w') as f:
            f.write(f'Probability of I2 populating n=2 vibrational level at {T} K: {probability_n2:.6e}\n')
    except IOError as e:
        print(f'Error writing to file: {e}')
