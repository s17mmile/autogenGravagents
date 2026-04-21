# filename: calculate_mortgage_payment.py

def calculate_monthly_payment(principal, annual_rate, years):
    """
    Calculate the monthly mortgage payment using the amortization formula.

    Parameters:
    principal (float): The loan amount.
    annual_rate (float): The annual interest rate as a decimal (e.g., 0.09 for 9%).
    years (int): The loan term in years.

    Returns:
    float: The monthly payment amount.

    Raises:
    ValueError: If principal, annual_rate, or years are not positive.
    """
    if principal <= 0:
        raise ValueError("Principal must be positive.")
    if annual_rate <= 0:
        raise ValueError("Annual interest rate must be positive and provided as a decimal.")
    if years <= 0:
        raise ValueError("Loan term in years must be positive.")

    monthly_rate = annual_rate / 12
    total_payments = years * 12
    numerator = monthly_rate * (1 + monthly_rate) ** total_payments
    denominator = (1 + monthly_rate) ** total_payments - 1
    monthly_payment = principal * numerator / denominator
    return monthly_payment

# Given values
principal = 100000
annual_interest_rate = 0.09  # 9% as decimal
loan_years = 30

monthly_payment = calculate_monthly_payment(principal, annual_interest_rate, loan_years)

print(f"Monthly mortgage payment: ${monthly_payment:.2f}")
