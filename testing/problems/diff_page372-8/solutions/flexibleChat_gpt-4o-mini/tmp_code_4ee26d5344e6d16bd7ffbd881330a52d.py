import numpy as np

def compute_inner_product(vector):
    """
    Computes the inner product of a complex vector with itself.
    
    Parameters:
    vector (np.array): A complex vector.
    
    Returns:
    complex: The inner product of the vector with itself.
    """
    return np.vdot(vector, vector)

# Define the vector y
vector_y = np.array([-1 + 1j, 2, 3 - 1j])

# Compute the inner product (y, y)
inner_product = compute_inner_product(vector_y)

# Print the result
print(f'The inner product (y, y) is: {inner_product}')