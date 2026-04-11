def compute_final_pressure(P0_bar: float, T0_C: float, T_final_C: float, beta_water: float, beta_vessel: float, kappa_water: float) -> tuple[float, float]:
    """
    Compute final pressure for a sealed, thermally expanding vessel filled with water.

    Parameters:
    - P0_bar: initial (absolute) pressure in bar
    - T0_C: initial temperature in Celsius
    - T_final_C: final temperature in Celsius
    - beta_water: volumetric thermal expansion coefficient of water (1/K)
    - beta_vessel: volumetric thermal expansion coefficient of the vessel (1/K)
    - kappa_water: isothermal compressibility of water (1/bar)

    Returns:
    - delta_P_bar: pressure rise in bar (absolute increase)
    - P_final_bar: final absolute pressure in bar
    """
    if kappa_water <= 0:
        raise ValueError("kappa_water must be positive.")
    if not all(isinstance(v, (int, float)) for v in (P0_bar, T0_C, T_final_C, beta_water, beta_vessel, kappa_water)):
        raise TypeError("All inputs must be numeric.")

    delta_T = T_final_C - T0_C  # delta_T in Kelvin; for heating, ΔT in K equals ΔT in C
    delta_P_bar = (beta_water - beta_vessel) * delta_T / kappa_water
    P_final_bar = P0_bar + delta_P_bar
    return delta_P_bar, P_final_bar


if __name__ == "__main__":
    P0_bar = 1.00
    T0_C = 13.56
    T_final_C = 82.0
    beta_water = 2.04e-4
    beta_vessel = 1.42e-4
    kappa_water = 4.59e-5

    delta_P_bar, P_final_bar = compute_final_pressure(P0_bar, T0_C, T_final_C, beta_water, beta_vessel, kappa_water)

    print("DeltaT = {:.2f} K".format(T_final_C - T0_C))
    print("DeltaP = {:.3f} bar".format(delta_P_bar))
    print("Final absolute pressure P_final = {:.3f} bar".format(P_final_bar))
    print("Gauge pressure (relative to ambient) = {:.3f} bar".format(delta_P_bar))