import math

# Constants
SCALE_PARAMETER = 10
NUM_COMPONENTS = 3
TIME_PERIOD = 1  # in hours

def calculate_failure_probability():
    """
    Calculate the probability that the device fails in the first hour of operation.
    The device fails if at least one of its components fails within the specified time period.
    """
    # Calculate the survival probability for one component
    survival_probability = math.exp(-TIME_PERIOD**2 / (SCALE_PARAMETER ** 2))
    
    # Calculate the probability that all components survive the first hour
    all_survive_probability = survival_probability ** NUM_COMPONENTS
    
    # Calculate the probability that at least one component fails in the first hour
    failure_probability = 1 - all_survive_probability
    
    # Print the result
    print(f'Probability that the device fails in the first hour: {failure_probability:.4f} (or {failure_probability * 100:.2f}%)')

# Execute the function
calculate_failure_probability()