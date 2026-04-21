# filename: vdw_pressure_ethane.py

def calculate_vdw_pressure(n, V, T_celsius, a, b, R=0.08206):
    """
    Calculate pressure using van der Waals equation of state.
    Parameters:
        n (float): number of moles (mol)
        V (float): volume in liters (L)
        T_celsius (float): temperature in degrees Celsius
        a (float): van der Waals constant a (L^2 atm / mol^2)
        b (float): van der Waals constant b (L / mol)
        R (float): ideal gas constant (default 0.08206 L atm / mol K)
    Returns:
        float: pressure in atm
    Raises:
        ValueError: if volume is not greater than n*b
    """
    if V <= n * b:
        raise ValueError(f"Volume must be greater than n*b to avoid non-physical results. Given V={V}, n*b={n*b}")
    T = T_celsius + 273.15  # convert to Kelvin
    P = (n * R * T) / (V - n * b) - a * (n / V)**2
    return P


# Given values for ethane
n = 10.0  # mol
V = 4.860  # L (dm^3)
T_celsius = 27.0  # degrees Celsius

a = 5.507  # L^2 atm / mol^2
b = 0.0651  # L / mol

pressure = calculate_vdw_pressure(n, V, T_celsius, a, b)
print(f"Pressure exerted by ethane: {pressure:.3f} atm")
