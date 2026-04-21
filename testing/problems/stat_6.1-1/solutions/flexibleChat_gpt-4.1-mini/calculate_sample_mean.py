# filename: calculate_sample_mean.py

def calculate_sample_mean(times):
    if not times:
        raise ValueError("The list of times must not be empty.")
    return sum(times) / len(times)

# Given sample times in seconds
sample_times = [1.1, 0.9, 1.4, 1.1, 1.0]

# Calculate sample mean
sample_mean = calculate_sample_mean(sample_times)

# Print the result
print(f"Sample mean (x-bar): {sample_mean:.3f} seconds")
