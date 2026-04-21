# filename: rocket_liftoff_time.py

def calculate_liftoff_time(m0, burn_rate, exhaust_velocity, g=9.81):
    """Calculate the time after firing when the rocket will lift off.

    Parameters:
    m0 (float): Initial mass of the rocket in kilograms.
    burn_rate (float): Fuel burn rate in kilograms per second.
    exhaust_velocity (float): Exhaust velocity in meters per second.
    g (float): Gravitational acceleration in m/s^2 (default 9.81).

    Returns:
    float: Time in seconds after firing when the rocket lifts off. Zero if immediate.
    """
    # Input validation
    if m0 <= 0 or burn_rate <= 0 or exhaust_velocity <= 0 or g <= 0:
        raise ValueError("All input parameters must be positive numbers.")

    thrust = burn_rate * exhaust_velocity
    weight_initial = m0 * g

    if thrust > weight_initial:
        return 0.0  # Immediate lift-off
    else:
        t = (m0 * g - thrust) / (burn_rate * g)
        if t < 0:
            return 0.0
        return t

# Given values
initial_mass = 7e4  # kg
fuel_burn_rate = 250  # kg/s
exhaust_velocity = 2500  # m/s

try:
    time_to_liftoff = calculate_liftoff_time(initial_mass, fuel_burn_rate, exhaust_velocity)
    # Save the result to a file
    with open('liftoff_time.txt', 'w') as f:
        f.write(f'Time to liftoff after firing (seconds): {time_to_liftoff:.2f}\n')
except Exception as e:
    with open('liftoff_time.txt', 'w') as f:
        f.write(f'Error calculating liftoff time: {e}\n')
