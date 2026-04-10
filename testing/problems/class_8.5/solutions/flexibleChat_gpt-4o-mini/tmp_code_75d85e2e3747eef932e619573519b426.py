import math

# Constants
AU = 1.496e11  # Astronomical Unit in meters
mu = 1.327e20  # Standard gravitational parameter for the Sun in m^3/s^2

def calculate_hohmann_transfer_time():
    # Distances from the Sun
    r1 = AU  # Distance from the Sun to Earth in meters
    r2 = 1.52 * AU  # Distance from the Sun to Mars in meters

    # 1. Calculate the semi-major axis of the Hohmann transfer orbit
    semi_major_axis_m = (r1 + r2) / 2

    # 2. Calculate the time of flight for the Hohmann transfer
    # Time of flight in seconds
    transfer_time_seconds = 0.5 * math.sqrt((semi_major_axis_m ** 3) / mu)

    # Convert time from seconds to days
    transfer_time_days = transfer_time_seconds / (60 * 60 * 24)  # 60 seconds/minute, 60 minutes/hour, 24 hours/day

    # Output the result
    print(f'Time needed for Hohmann transfer: {transfer_time_days:.2f} days')

# Call the function to execute the calculation
calculate_hohmann_transfer_time()