import math

# Given values
W_at_t2 = 3  # Wronskian at t = 2

# Define the Wronskian function
# This function calculates the Wronskian at a given t value based on the constant C.
def W(t, C):
    return C * math.exp(1 - 2 / t)

# Main execution logic
if __name__ == '__main__':
    t_value_2 = 2  # t value for W_2
    t_value_4 = 4  # t value for W_4

    # Calculate the constant C
    C = W_at_t2 * math.exp(2 / t_value_2)

    # Calculate W(4)
    W_at_t4 = W(t_value_4, C)

    # Output the result
    print(f'W(y1, y2)(4) = {W_at_t4}')