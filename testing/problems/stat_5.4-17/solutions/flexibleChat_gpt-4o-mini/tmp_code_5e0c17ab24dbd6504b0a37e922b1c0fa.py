import numpy as np

def simulate_dropout_probability(n=25, p=0.5, num_simulations=100000, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducibility
    
    # Simulate dropouts for both groups
    X = np.random.binomial(n, p, num_simulations)  # dropouts from old treatment
    Y = np.random.binomial(n, p, num_simulations)  # dropouts from new treatment
    
    # Calculate the probability that Y >= X + 2
    probability = np.mean(Y >= X + 2)
    
    # Output the result
    print(f'Probability that Y >= X + 2 (based on {num_simulations} simulations): {probability}')  

# Run the simulation
simulate_dropout_probability(seed=42)  # Example with a seed for reproducibility