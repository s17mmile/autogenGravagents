# filename: verify_line_integral.py
import numpy as np

# Parametrize the boundary circle: r(t) = (cos t, sin t, 0), t in [0, 2*pi]
# Vector field A = (y, z, x) = (sin t, 0, cos t) on boundary

def A(t):
    return np.array([np.sin(t), 0, np.cos(t)])

def dr_dt(t):
    return np.array([-np.sin(t), np.cos(t), 0])

# Compute the line integral numerically using trapezoidal rule
N = 10000
t = np.linspace(0, 2*np.pi, N)
A_vals = np.array([A(ti) for ti in t])
dr_vals = np.array([dr_dt(ti) for ti in t])

integrand = np.einsum('ij,ij->i', A_vals, dr_vals)  # dot product at each t

line_integral = np.trapz(integrand, t)

print(f'Numerical line integral of A over boundary circle: {line_integral}')

with open('line_integral_result.txt', 'w') as f:
    f.write(f'Numerical line integral of A over boundary circle: {line_integral}\n')
