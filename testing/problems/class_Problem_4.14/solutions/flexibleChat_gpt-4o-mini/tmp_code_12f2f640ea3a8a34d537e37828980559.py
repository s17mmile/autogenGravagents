def find_divergence(alpha, x1, x2):
    n = 1  # Start counting iterations from 1
    while n <= 1000:  # Prevent infinite loop
        x1 = alpha * x1 * (1 - x1 ** 2)
        x2 = alpha * x2 * (1 - x2 ** 2)
        if abs(x1 - x2) > 0.3 * min(x1, x2):
            return n
        n += 1
    return None  # Return None if divergence condition is not met within 1000 iterations

alpha = 2.5
x1 = 0.9000000
x2 = 0.9000001
n = find_divergence(alpha, x1, x2)
if n is not None:
    print(f'The lowest value of n for which the two values diverge by more than 30% is: {n}')
else:
    print('Divergence condition not met within 1000 iterations.')