from typing import Optional, Dict


def lifetime_bounds(tau: float, central_wavelength_nm: Optional[float] = None) -> Dict[str, float]:
    """
    Compute energy and frequency bounds for lifetime broadening.

    Parameters:
    - tau: lifetime in seconds (must be > 0).
    - central_wavelength_nm: optional central wavelength in nanometers; if provided,
      will also compute a corresponding wavelength broadening using the hbar/tau bound.

    Returns a dictionary with values in SI units (J, Hz) and also in eV for DeltaE.
    """
    if tau <= 0:
        raise ValueError("tau must be positive and non-zero")

    # Physical constants (SI)
    h = 6.62607015e-34          # Planck constant, J*s
    hbar = 1.0545718e-34        # Reduced Planck constant, J*s
    c = 299792458.0               # Speed of light, m/s
    J_per_eV = 1.602176634e-19     # J per eV

    # Three conventional bounds
    delta_E_hbar_J = hbar / tau          # DeltaE ~ hbar/tau
    delta_E_hbar2_J = hbar / (2.0 * tau) # DeltaE ~ hbar/(2*tau)
    delta_E_h_J = h / tau                 # DeltaE ~ h/tau

    delta_nu_hbar_Hz = delta_E_hbar_J / h
    delta_nu_hbar2_Hz = delta_E_hbar2_J / h
    delta_nu_h_Hz = delta_E_h_J / h

    delta_E_hbar_eV = delta_E_hbar_J / J_per_eV
    delta_E_hbar2_eV = delta_E_hbar2_J / J_per_eV
    delta_E_h_eV = delta_E_h_J / J_per_eV

    results: Dict[str, float] = {
        'tau_s': tau,
        'DeltaE_hbar_J': delta_E_hbar_J,
        'DeltaE_hbar_eV': delta_E_hbar_eV,
        'DeltaNu_hbar_Hz': delta_nu_hbar_Hz,
        'DeltaE_hbar2_J': delta_E_hbar2_J,
        'DeltaE_hbar2_eV': delta_E_hbar2_eV,
        'DeltaNu_hbar2_Hz': delta_nu_hbar2_Hz,
        'DeltaE_h_J': delta_E_h_J,
        'DeltaE_h_eV': delta_E_h_eV,
        'DeltaNu_h_Hz': delta_nu_h_Hz,
    }

    # Optional wavelength-related calculation
    if central_wavelength_nm is not None:
        lam = central_wavelength_nm * 1e-9  # convert nm to meters
        # Using DeltaE_hbar_J to compute DeltaLambda (approx): dE = -(h c / lambda^2) dlambda
        delta_lambda_m = (lam**2 / (h * c)) * delta_E_hbar_J
        delta_lambda_nm = delta_lambda_m * 1e9
        results['DeltaLambda_using_hbar_nm'] = delta_lambda_nm

    return results


def main():
    tau = 1e-9  # 1 nanosecond
    # Example without central wavelength
    res = lifetime_bounds(tau)
    print(f"tau = {tau:.3e} s")
    print(f"DeltaE (hbar/tau) = {res['DeltaE_hbar_J']:.3e} J, {res['DeltaE_hbar_eV']:.3e} eV; DeltaNu = {res['DeltaNu_hbar_Hz']:.3e} Hz")
    print(f"DeltaE (hbar/(2*tau)) = {res['DeltaE_hbar2_J']:.3e} J, {res['DeltaE_hbar2_eV']:.3e} eV; DeltaNu = {res['DeltaNu_hbar2_Hz']:.3e} Hz")
    print(f"DeltaE (h/tau) = {res['DeltaE_h_J']:.3e} J, {res['DeltaE_h_eV']:.3e} eV; DeltaNu = {res['DeltaNu_h_Hz']:.3e} Hz")

    # Example with central wavelength to illustrate DeltaLambda calculation
    central_lambda_nm = 500.0  # example: 500 nm center
    res_with_lambda = lifetime_bounds(tau, central_wavelength_nm=central_lambda_nm)
    if 'DeltaLambda_using_hbar_nm' in res_with_lambda:
        print(f"DeltaLambda_using_hbar_nm (for lambda = {central_lambda_nm} nm) = {res_with_lambda['DeltaLambda_using_hbar_nm']:.3e} nm")


if __name__ == "__main__":
    main()
