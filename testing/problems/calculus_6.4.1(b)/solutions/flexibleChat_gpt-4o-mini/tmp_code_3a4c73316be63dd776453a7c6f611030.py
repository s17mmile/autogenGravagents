def calculate_work(weight, height):
    """Calculate the work done against gravity."""
    return weight * height

# Constants
weight_of_gorilla_lb = 360  # weight in pounds
height_climbed_ft = 20       # height in feet

# Work done calculation
work_done_ft_lb = calculate_work(weight_of_gorilla_lb, height_climbed_ft)

# Output the result
print(f'The work done by the gorilla in climbing the tree is {work_done_ft_lb} ft·lb.')