# filename: read_burnout_speed.py

# Read the rocket burnout speed from the output file and print it
with open('rocket_burnout_speed.txt', 'r') as f:
    burnout_speed = f.read().strip()

print(burnout_speed)
