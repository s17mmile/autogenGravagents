# filename: point_estimate_mu.py

def calculate_sample_mean(data):
    if not data:
        raise ValueError("Data list is empty. Cannot compute mean.")
    return sum(data) / len(data)

# Given data on pea plant growth
pea_growth_data = [17.5, 14.5, 15.2, 14.0, 17.3, 18.0, 13.8]

# Calculate the point estimate of mu (sample mean)
point_estimate_mu = calculate_sample_mean(pea_growth_data)

# Print the result
print(f"Point estimate of mu (sample mean): {point_estimate_mu:.2f}")
