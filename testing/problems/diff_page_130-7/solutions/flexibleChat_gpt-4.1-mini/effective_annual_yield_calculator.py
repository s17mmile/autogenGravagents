# filename: effective_annual_yield_calculator.py

def calculate_effective_annual_yield(annual_rate, compounding_periods):
    """
    Calculate the effective annual yield (EAY) given an annual nominal interest rate
    and the number of compounding periods per year.

    Parameters:
    annual_rate (float): The nominal annual interest rate as a decimal (e.g., 0.07 for 7%).
    compounding_periods (int): Number of compounding periods per year (e.g., 365 for daily).

    Returns:
    float: The effective annual yield as a decimal.
    """
    if annual_rate < 0:
        raise ValueError("Annual interest rate must be non-negative.")
    if compounding_periods <= 0:
        raise ValueError("Compounding periods must be a positive integer.")

    periodic_rate = annual_rate / compounding_periods
    eay = (1 + periodic_rate) ** compounding_periods - 1
    return eay


# Example usage
if __name__ == "__main__":
    annual_nominal_rate = 0.07  # 7% annual interest rate
    compounding_periods_per_year = 365  # daily compounding

    effective_annual_yield = calculate_effective_annual_yield(annual_nominal_rate, compounding_periods_per_year)
    print(f"Effective Annual Yield (EAY): {effective_annual_yield:.6f}")
