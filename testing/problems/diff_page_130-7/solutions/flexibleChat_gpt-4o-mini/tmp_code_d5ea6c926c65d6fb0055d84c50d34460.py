def calculate_effective_annual_yield(initial_balance, annual_interest_rate, compounding_frequency, investment_duration):
    # Error handling for input values
    if initial_balance <= 0:
        raise ValueError('Initial balance must be greater than zero.')
    if not (0 <= annual_interest_rate <= 1):
        raise ValueError('Annual interest rate must be between 0 and 1.')

    # Calculate final balance using compound interest formula
    final_balance = initial_balance * (1 + annual_interest_rate / compounding_frequency) ** (compounding_frequency * investment_duration)

    # Calculate effective annual yield
    effective_annual_yield = (final_balance - initial_balance) / initial_balance

    # Format output for better readability
    return round(final_balance, 2), round(effective_annual_yield * 100, 2)  # Yield as percentage

# Example usage
initial_balance = 1000  # Example initial balance
annual_interest_rate = 0.07  # 7% interest rate
compounding_frequency = 365  # Daily compounding
investment_duration = 1  # 1 year

final_balance, effective_annual_yield = calculate_effective_annual_yield(initial_balance, annual_interest_rate, compounding_frequency, investment_duration)

# Output results
print(f'Final Balance: ${final_balance}')
print(f'Effective Annual Yield: {effective_annual_yield}%')