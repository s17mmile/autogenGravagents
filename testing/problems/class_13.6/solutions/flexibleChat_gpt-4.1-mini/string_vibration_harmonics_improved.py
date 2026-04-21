# filename: string_vibration_harmonics_improved.py
import sympy as sp

# Define symbols
L, Vmax, n = sp.symbols('L Vmax n', real=True, positive=True)
x = sp.symbols('x', real=True)

# Wave speed c (will cancel out in ratio)
c = sp.symbols('c', real=True, positive=True)

# Angular frequency for nth mode
omega_n = n * sp.pi * c / L

# Initial velocity v(x,0) as a piecewise triangular function:
# - Increases linearly from 0 at x=0 to max at x=L/4
# - Decreases linearly from max at x=L/4 to 0 at x=L/2
# - Zero for x > L/2
v = sp.Piecewise(
    ((4*Vmax/L)*x, (x >= 0) & (x <= L/4)),
    ((4*Vmax/L)*(L/2 - x), (x > L/4) & (x <= L/2)),
    (0, True)
)

# The string has fixed ends at x=0 and x=L,
# so normal modes are sin(n*pi*x/L).

# Compute B_n coefficients from initial velocity:
# B_n = (2/(L*omega_n)) * integral_0^L v(x,0)*sin(n*pi*x/L) dx
# Since v=0 for x > L/2, integral limits reduce to 0 to L/2
sin_term = sp.sin(n * sp.pi * x / L)

integral_expr = sp.integrate(v * sin_term, (x, 0, L/2))

B_n_expr = (2 / (L * omega_n)) * integral_expr

# Substitute omega_n expression
B_n_expr = B_n_expr.subs(omega_n, n * sp.pi * c / L)

# Simplify expression
B_n_expr = sp.simplify(B_n_expr)

# Compute B_1 and B_2 explicitly
B_1 = B_n_expr.subs(n, 1)
B_2 = B_n_expr.subs(n, 2)

# Compute ratio R = B_2 / B_1
# Note: wave speed c cancels out in ratio
R = sp.simplify(B_2 / B_1)

# Compute decibel difference: 20*log10(|R|)
dB_diff = 20 * sp.log(sp.Abs(R), 10)


def numeric_decibel_difference(Vmax_val=1, L_val=1, c_val=1):
    """
    Numerically evaluate the decibel difference between second harmonic and fundamental.
    Parameters:
        Vmax_val: Maximum initial velocity amplitude
        L_val: Length of the string
        c_val: Wave speed (cancels out in ratio, default=1)
    Returns:
        Decibel difference as float
    """
    subs_dict = {Vmax: Vmax_val, L: L_val, c: c_val}
    return float(dB_diff.evalf(subs=subs_dict))

# Example numeric evaluation
decibel_difference = numeric_decibel_difference()

# Output symbolic expressions and numeric decibel difference
print("Symbolic B_1 coefficient:", B_1)
print("Symbolic B_2 coefficient:", B_2)
print(f"Decibel difference between second harmonic and fundamental: {decibel_difference:.2f} dB")

# The subsequent motion of the string is given by:
# y(x,t) = sum_{n=1}^\infty B_n sin(n pi x / L) sin(omega_n t)
# with B_n as above and omega_n = n pi c / L

# This code snippet provides the coefficients and decibel difference as requested.