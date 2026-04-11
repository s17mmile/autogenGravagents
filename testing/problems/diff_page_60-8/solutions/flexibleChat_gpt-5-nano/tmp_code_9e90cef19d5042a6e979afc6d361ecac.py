import math

# Default tolerance for floating point comparisons
_TOL = 1e-12


def required_continuous_deposit(F: float, r: float, T: float, tol: float = _TOL) -> float:
    """Return the required annual continuous deposit k to reach future value F with
    continuous compounding at rate r over time T.

    Model:
        A(T) = (k / r) * (exp(r*T) - 1) for r > 0
        A(T) = k * T for r = 0

    Precondition: F > 0, T > 0, r >= 0
    tol: tolerance to treat near-zero values for r or denom
    """
    if F <= 0 or T <= 0:
        raise ValueError("F and T must be positive.")
    if r < 0:
        raise ValueError("r must be non-negative in this simplified model.")

    # Handle near-zero interest rate gracefully
    if abs(r) < tol:
        return F / T

    # Compute denominator with stability; guard against near-zero denom and overflow
    try:
        denom = math.expm1(r * T)  # exp(r*T) - 1
    except OverflowError as e:
        raise OverflowError(
            f"Overflow during expm1 computation for r*T = {r*T}. Parameters may be too large for float representation."
        ) from e

    if abs(denom) < tol:
        raise ValueError("Denominator evaluated to near zero; adjust parameters.")

    return F * r / denom


def main():
    F = 1_000_000
    r = 0.075
    T = 40
    k = required_continuous_deposit(F, r, T)
    print("Required annual continuous deposit k: ${:,.2f}".format(k))


if __name__ == "__main__":
    main()
