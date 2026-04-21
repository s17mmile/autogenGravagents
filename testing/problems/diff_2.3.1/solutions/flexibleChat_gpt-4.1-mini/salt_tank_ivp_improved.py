# filename: salt_tank_ivp_improved.py
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def solve_salt_tank_ode(r_val, Q0_val):
    """Solve the initial value problem for salt amount Q(t) in the tank."""
    # Define symbols
    Q = sp.Function('Q')
    t = sp.symbols('t', real=True, positive=True)

    # Define the ODE: dQ/dt = r/4 - (r/100)*Q
    ode = sp.Eq(Q(t).diff(t), r_val/4 - (r_val/100)*Q(t))

    # Initial condition
    ics = {Q(0): Q0_val}

    # Solve the ODE
    Q_sol = sp.dsolve(ode, Q(t), ics=ics)

    # Extract solution expression
    Q_t = Q_sol.rhs

    return t, Q_t


def compute_limiting_amount(Q_t, t):
    """Compute the limiting amount Q_L as t approaches infinity."""
    Q_L = sp.limit(Q_t, t, sp.oo)
    return Q_L


def compute_time_to_within_percentage(Q_t, Q_L_val, Q0_val, r_val, percentage=0.02):
    """Compute time T such that |Q(t) - Q_L| <= percentage * Q_L."""
    lhs = abs(Q0_val - Q_L_val)
    rate = r_val / 100
    threshold = percentage * Q_L_val

    # Check for valid input to logarithm
    if threshold <= 0 or lhs <= 0:
        raise ValueError("Threshold and initial difference must be positive for logarithm calculation.")

    T_val = -np.log(threshold / lhs) / rate
    return T_val


def main():
    # Given parameters
    r_val = 3  # gal/min

    # Limiting amount Q_L is independent of Q0, so first solve with symbolic Q0
    Q0_sym = sp.symbols('Q0', real=True, positive=True)
    t, Q_t_sym = solve_salt_tank_ode(r_val, Q0_sym)
    Q_L_sym = compute_limiting_amount(Q_t_sym, t)

    # Evaluate Q_L numerically
    Q_L_val = float(Q_L_sym.evalf())

    # Given initial amount Q0 = 2 * Q_L
    Q0_val = 2 * Q_L_val

    # Solve ODE with numeric Q0
    t, Q_t = solve_salt_tank_ode(r_val, Q0_val)

    # Compute time T to be within 2% of Q_L
    T_val = compute_time_to_within_percentage(Q_t, Q_L_val, Q0_val, r_val, percentage=0.02)

    # Print results
    print(f"Differential equation: dQ/dt = {r_val}/4 - ({r_val}/100)*Q")
    print(f"Initial condition: Q(0) = {Q0_val:.4f}")
    print(f"Limiting amount Q_L = {Q_L_val:.4f} lb")
    print(f"Time T to be within 2% of Q_L: {T_val:.2f} minutes")
    print(f"Solution Q(t) = {Q_t}")

    # Create numerical function for plotting
    Q_func = sp.lambdify(t, Q_t, 'numpy')

    # Plot Q(t) over time
    t_vals = np.linspace(0, T_val * 1.5, 300)
    Q_vals = Q_func(t_vals)

    plt.plot(t_vals, Q_vals, label='Q(t)')
    plt.axhline(Q_L_val, color='red', linestyle='--', label='Q_L')
    plt.axhline(Q_L_val * 1.02, color='green', linestyle=':', label='2% bounds')
    plt.axhline(Q_L_val * 0.98, color='green', linestyle=':')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Amount of salt Q(t) (lb)')
    plt.title('Salt amount in tank over time')
    plt.legend()
    plt.grid(True)
    plt.savefig('salt_tank_solution_plot.png')


if __name__ == '__main__':
    main()
