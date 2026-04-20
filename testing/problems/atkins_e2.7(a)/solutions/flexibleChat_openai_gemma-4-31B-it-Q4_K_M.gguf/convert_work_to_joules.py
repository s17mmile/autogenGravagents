# filename: convert_work_to_joules.py

# Given values
work_l_atm = -15.099480  # in L·atm
conversion_factor = 101.325  # J per L·atm

# Convert work to joules
work_joules = work_l_atm * conversion_factor

# Output the result with 3 decimal places for clarity
print(f"Work done by the system: {work_joules:.3f} J")