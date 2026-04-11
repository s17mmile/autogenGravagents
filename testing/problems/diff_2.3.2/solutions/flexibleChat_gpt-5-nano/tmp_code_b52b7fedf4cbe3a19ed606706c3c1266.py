import math
from typing import Dict


def futures_values(contribution_rate: float, growth_rate: float, years: int) -> Dict[str, float]:
    """Compute future values for continuous and discrete contributions.

    - continuous: c * (exp(r*T) - 1) / r  (continuous contributions and continuous compounding)
    - discrete: c * ((1 + r)**T - 1) / r  (annuity-immediate, deposits at year-end)

    If growth_rate == 0, return simple sums: contribution_rate * years for both.
    """
    if years <= 0:
        return {"fv_continuous": 0.0, "fv_discrete": 0.0}
    if growth_rate == 0:
        fv_continuous = contribution_rate * years
        fv_discrete = contribution_rate * years
    else:
        fv_continuous = contribution_rate * (math.exp(growth_rate * years) - 1.0) / growth_rate
        fv_discrete = contribution_rate * (((1.0 + growth_rate) ** years) - 1.0) / growth_rate
    return {
        "fv_continuous": fv_continuous,
        "fv_discrete": fv_discrete,
    }


def main():
    contribution_rate = 2000.0  # dollars per year, treated as a continuous contribution
    growth_rate = 0.08        # annual return, continuous compounding
    years = 40                # from age 25 to 65

    results = futures_values(contribution_rate, growth_rate, years)

    print(f"FV_continuous: ${results['fv_continuous']:,.2f}")
    print(f"FV_discrete: ${results['fv_discrete']:,.2f}")


if __name__ == "__main__":
    main()
