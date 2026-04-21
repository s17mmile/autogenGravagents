# filename: find_divergence_symmetric.py

def find_divergence(alpha, x1, x2, threshold=0.3, max_iter=1000):
    x_n_1 = x1
    x_n_2 = x2
    for n in range(1, max_iter + 1):
        x_n_1 = alpha * x_n_1 * (1 - x_n_1**2)
        x_n_2 = alpha * x_n_2 * (1 - x_n_2**2)
        # Calculate symmetric relative difference
        denom = (abs(x_n_1) + abs(x_n_2)) / 2
        if denom != 0:
            rel_diff = abs(x_n_1 - x_n_2) / denom
        else:
            rel_diff = abs(x_n_1 - x_n_2)
        if rel_diff > threshold:
            return n, x_n_1, x_n_2, rel_diff
    return None, x_n_1, x_n_2, rel_diff


alpha = 2.5
x1 = 0.9000000
x2 = 0.9000001
threshold = 0.3

result = find_divergence(alpha, x1, x2, threshold)

if result[0] is not None:
    n, val1, val2, diff = result
    print(f"Divergence exceeds 30% at iteration n = {n}")
    print(f"x_n starting from {x1} = {val1}")
    print(f"x_n starting from {x2} = {val2}")
    print(f"Relative difference = {diff:.4f}")
else:
    print("No divergence exceeding 30% found within the iteration limit.")
