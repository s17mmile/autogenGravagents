import math

def time_to_target(V, Q, Cin, A0, target_fraction):
    # Validate inputs
    if V <= 0 or Q <= 0:
        raise ValueError("V and Q must be positive.")
    if not (0 < target_fraction <= 1.0):
        raise ValueError("target_fraction must be between 0 and 1.")

    A_ss = Cin * V  # steady-state mass in the tank when inflow has concentration Cin
    A_target = A0 * target_fraction

    if A_target <= A_ss:
        raise ValueError("Target mass is not above steady-state (Cin * V); target unreachable with given Cin.")

    # A(t) = Cin*V + (A0 - Cin*V) * exp(-(Q/V) t)
    # Solve for t: t = (V/Q) * ln((A0 - Cin*V) / (A_target - Cin*V))
    t = (V / Q) * math.log((A0 - A_ss) / (A_target - A_ss))
    return t

def mass_and_concentration(V, Q, Cin, A0, t):
    A_t = Cin * V + (A0 - Cin * V) * math.exp(- (Q / V) * t)
    C_t = A_t / V
    return A_t, C_t

# Preset system parameters for the original problem (Cin = 0)
V = 200.0  # liters
Q = 2.0    # liters per minute
Cin = 0.0  # g per liter, fresh water inflow
C0 = 1.0   # g per liter, initial dye concentration
A0 = C0 * V  # initial mass of dye in grams
target_fraction = 0.01

# Compute time to reach the target and verify the state at that time
t_min = time_to_target(V, Q, Cin, A0, target_fraction)
A_t, C_t = mass_and_concentration(V, Q, Cin, A0, t_min)

print("Time to reach target (minutes):", t_min)
print("Time to reach target (hours):", t_min / 60.0)
print("Mass at target (g):", A_t)
print("Concentration at target (g/L):", C_t)
