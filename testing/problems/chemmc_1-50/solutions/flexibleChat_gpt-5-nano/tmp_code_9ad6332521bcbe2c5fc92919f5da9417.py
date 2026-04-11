from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundsResult:
    tau: float
    delta_E_hbar_J: float
    delta_E_hbar_eV: float
    delta_nu_hbar_Hz: float
    delta_E_hbar2_J: float
    delta_E_hbar2_eV: float
    delta_nu_hbar2_Hz: float
    delta_E_h_J: float
    delta_E_h_eV: float
    delta_nu_h_Hz: float
    delta_lambda_using_hbar_nm: Optional[float] = None


def compute_bounds(tau: float, central_wavelength_nm: Optional[float] = None, bound_for_lambda: str = 'hbar') -> BoundsResult:
    """
    Compute energy and frequency bounds for lifetime broadening.

    Parameters:
    - tau: lifetime in seconds (must be > 0)
    - central_wavelength_nm: optional central wavelength in nanometers; if provided,
      compute DeltaLambda using the selected bound_for_lambda (default 'hbar').
    - bound_for_lambda: which bound to use for the DeltaLambda calculation ('hbar', 'hbar2', 'h')

    Returns a BoundsResult data object with all computed quantities.
    """
    if tau <= 0:
        raise ValueError("tau must be positive and non-zero")

    # Physical constants (SI)
    h = 6.62607015e-34          # Planck constant, J*s
    hbar = 1.0545718e-34        # Reduced Planck constant, J*s
    c = 299792458.0               # Speed of light, m/s
    J_per_eV = 1.602176634e-19   # J per eV

    # Three conventional bounds
    delta_E_hbar_J = hbar / tau                # DeltaE ~ hbar/tau
    delta_E_hbar2_J = hbar / (2.0 * tau)       # DeltaE ~ hbar/(2*tau)
    delta_E_h_J = h / tau                       # DeltaE ~ h/tau

    delta_nu_hbar_Hz = delta_E_hbar_J / h        # DeltaNu corresponding to DeltaE_hbar
    delta_nu_hbar2_Hz = delta_E_hbar2_J / h      # DeltaNu for DeltaE_hbar2
    delta_nu_h_Hz = delta_E_h_J / h               # DeltaNu for DeltaE_h

    delta_E_hbar_eV = delta_E_hbar_J / J_per_eV
    delta_E_hbar2_eV = delta_E_hbar2_J / J_per_eV
    delta_E_h_eV = delta_E_h_J / J_per_eV

    delta_lambda_using_hbar_nm = None
    if central_wavelength_nm is not None and bound_for_lambda == 'hbar':
        lam = central_wavelength_nm * 1e-9  # convert nm to meters
        delta_lambda_using_hbar = (lam**2 / (h * c)) * delta_E_hbar_J
        delta_lambda_using_hbar_nm = delta_lambda_using_hbar * 1e9

    return BoundsResult(
        tau=tau,
        delta_E_hbar_J=delta_E_hbar_J,
        delta_E_hbar_eV=delta_E_hbar_eV,
        delta_nu_hbar_Hz=delta_nu_hbar_Hz,
        delta_E_hbar2_J=delta_E_hbar2_J,
        delta_E_hbar2_eV=delta_E_hbar2_eV,
        delta_nu_hbar2_Hz=delta_nu_hbar2_Hz,
        delta_E_h_J=delta_E_h_J,
        delta_E_h_eV=delta_E_h_eV,
        delta_nu_h_Hz=delta_nu_h_Hz,
        delta_lambda_using_hbar_nm=delta_lambda_using_hbar_nm,
    )


def format_bounds_table(r: BoundsResult) -> str:
    lines = []
    lines.append(f"tau = {r.tau:.3e} s")
    lines.append("")
    lines.append("{:<20}{:>15}{:>15}{:>15}".format('Bound','DeltaE_J','DeltaE_eV','DeltaNu_Hz'))
    lines.append("{:<20}{:>15.6e}{:>15.6e}{:>15.6e}".format(
        'DeltaE_hbar', r.delta_E_hbar_J, r.delta_E_hbar_eV, r.delta_nu_hbar_Hz))
    lines.append("{:<20}{:>15.6e}{:>15.6e}{:>15.6e}".format(
        'DeltaE_hbar2', r.delta_E_hbar2_J, r.delta_E_hbar2_eV, r.delta_nu_hbar2_Hz))
    lines.append("{:<20}{:>15.6e}{:>15.6e}{:>15.6e}".format(
        'DeltaE', r.delta_E_h_J, r.delta_E_h_eV, r.delta_nu_h_Hz))
    if r.delta_lambda_using_hbar_nm is not None:
        lines.append(f"DeltaLambda_using_hbar_nm: {r.delta_lambda_using_hbar_nm:.6e} nm")
    return "\n".join(lines)


def main():
    tau = 1e-9
    res = compute_bounds(tau, central_wavelength_nm=500.0, bound_for_lambda='hbar')
    print(format_bounds_table(res))


if __name__ == "__main__":
    main()
