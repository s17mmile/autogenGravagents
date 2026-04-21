# filename: calculate_account_balance_function.py

def calculate_balance(initial_deposit, annual_interest_rate, monthly_deposit, years):
    """
    Calculate the future balance of an account with an initial deposit,
    monthly compounding interest, and additional monthly deposits.

    Parameters:
    initial_deposit (float): The initial amount deposited.
    annual_interest_rate (float): Annual interest rate as a decimal (e.g., 0.08 for 8%).
    monthly_deposit (float): The amount deposited at the end of each month.
    years (int): Number of years the money is invested.

    Returns:
    float: The balance after the specified number of years.
    """
    months = years * 12
    monthly_interest_rate = annual_interest_rate / 12

    # Future value of the initial deposit compounded monthly
    fv_initial = initial_deposit * (1 + monthly_interest_rate) ** months

    # Future value of the series of monthly deposits (ordinary annuity formula)
    fv_deposits = monthly_deposit * (((1 + monthly_interest_rate) ** months - 1) / monthly_interest_rate)

    # Total future value
    fv_total = fv_initial + fv_deposits
    return fv_total


# Given values
initial_deposit = 1000
annual_interest_rate = 0.08
monthly_deposit = 25
years = 3

# Calculate balance
balance_after_3_years = calculate_balance(initial_deposit, annual_interest_rate, monthly_deposit, years)

# Print the result
print(f"Balance after {years} years: ${balance_after_3_years:.2f}")
