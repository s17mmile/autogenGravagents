import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constant for the trajectory equation
GRAVITY_CONSTANT = 45

# Define the derivative of y with respect to x
def derivative_of_y(x):
    return -2 * x / GRAVITY_CONSTANT

# Define the integrand for the arc length formula
def arc_length_integrand(x):
    return np.sqrt(1 + (derivative_of_y(x))**2)

# Function to plot the integrand
def plot_integrand():
    x_values = np.linspace(0, 90, 100)
    y_values = arc_length_integrand(x_values)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, label='Integrand for Arc Length', color='blue')
    plt.title('Integrand Behavior Over Integration Range')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Integrand Value')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.legend()
    plt.grid()
    plt.show()

# Function to calculate the distance traveled by the prey

def calculate_distance():
    x_start = 0
    x_end = 90
    arc_length, _ = quad(arc_length_integrand, x_start, x_end)
    return round(arc_length, 1)

# Execute the plotting and calculation
if __name__ == '__main__':
    plot_integrand()  # Visualize the integrand
    distance = calculate_distance()  # Calculate the distance
    print(f'Distance traveled by the prey: {distance} m')