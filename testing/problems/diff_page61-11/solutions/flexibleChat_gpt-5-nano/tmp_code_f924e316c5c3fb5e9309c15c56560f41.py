from typing import Tuple

def payoff_month(
    initial_balance: float = 100000.0,
    annual_rate: float = 0.09,
    initial_payment: float = 800.0,
    growth_per_month: float = 1.0 / 120.0,
    epsilon: float = 1e-12,
    max_months: int = 20000,
) -> Tuple[int, float]:
    """
    Compute when the loan is fully paid under an end-of-month payment schedule.

    Payments: P_t = initial_payment * (1 + t/120) where t is the 1-based month index.
    Balance evolution: B_t = r * B_{t-1} - P_t with r = 1 + annual_rate/12.

    Returns:
        (months_to_payoff, final_payment)
    """
    i = annual_rate / 12.0
    r = 1.0 + i
    balance = float(initial_balance)
    month = 0

    while month < max_months:
        month += 1
        payment = initial_payment * (1.0 + month * growth_per_month)
        balance_before_payment = balance * r

        if balance_before_payment <= payment + epsilon:
            final_payment = max(0.0, float(balance_before_payment))
            return month, final_payment

        balance = balance_before_payment - payment

    raise RuntimeError("Payoff not achieved within the maximum allowed months.")

# Example usage (no user input):
months, final_payment = payoff_month()
print("Months until payoff:", months)
print("Final (partial) payment:", round(final_payment, 2))