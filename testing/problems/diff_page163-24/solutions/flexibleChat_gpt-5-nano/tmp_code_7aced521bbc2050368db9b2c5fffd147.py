import numpy as np

# Problem data (pre-computed constants)
C = 2.0
D = 7.0/np.sqrt(34.0)
beta = np.sqrt(34.0)/5.0
phi = np.arctan2(D, C)
R = np.sqrt(C**2 + D**2)

def u(t):
    # Analytic solution: u(t) = exp(-t/5) [C cos(beta t) + D sin(beta t)]
    return np.exp(-t/5.0) * (C*np.cos(beta*t) + D*np.sin(beta*t))

def t_peak(k):
    # Peaks occur at beta t - phi = k*pi  => t = (phi + k*pi)/beta
    return (phi + k*np.pi) / beta

def A_peak(k):
    # Amplitude at peak: |u(t_peak(k))| = exp(-t_peak(k)/5) * R
    return np.exp(-t_peak(k)/5.0) * R

# Simple brent-like root finder (bisection) for scalar functions
def brent_like_root(f, a, b, tol=1e-12, max_iter=200):
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa*fb > 0:
        return None
    for _ in range(max_iter):
        m = 0.5*(a + b)
        fm = f(m)
        if abs(fm) < 1e-14:
            return m
        if fa*fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if abs(b - a) < tol:
            return 0.5*(a + b)
    return 0.5*(a + b)

# Robust root finding in an interval by bracketing all sign changes for fpos and fneg

def find_roots_in_interval(a, b):
    # fpos(t) = u(t) - 0.1, fneg(t) = u(t) + 0.1
    def fpos(t): return u(t) - 0.1
    def fneg(t): return u(t) + 0.1
    roots = []
    N = 4000  # fine grid for robust bracketing
    for i in range(N):
        ta = a + (i / N) * (b - a)
        tb = a + ((i + 1) / N) * (b - a)
        fa, fb = fpos(ta), fpos(tb)
        if fa == 0:
            roots.append(ta)
        elif fa * fb <= 0:
            r = brent_like_root(fpos, ta, tb)
            if r is not None:
                roots.append(r)
        fa, fb = fneg(ta), fneg(tb)
        if fa == 0:
            roots.append(ta)
        elif fa * fb <= 0:
            r = brent_like_root(fneg, ta, tb)
            if r is not None:
                roots.append(r)
    return roots

# Determine the first index k0 where the peak amplitude drops below 0.1
k0 = None
# Look ahead a reasonable range of peaks; k up to 1000 is ample for this problem
for k in range(0, 1000):
    if A_peak(k) <= 0.1:
        k0 = k
        break

if k0 is None:
    # Fallback: envelope bound gives a safe, but not necessarily minimal, T
    A = R
    T_bound = 5.0 * np.log(10.0 * A)
    T_min = T_bound
else:
    # Define interval between last peak above threshold and the first peak below threshold
    left = max(0.0, t_peak(k0 - 1))  if k0 - 1 >= 0 else 0.0
    right = t_peak(k0)
    # Collect potential threshold-crossing times within (left, right)
    roots = find_roots_in_interval(left, right)
    if len(roots) == 0:
        # If no root found numerically, fall back to the right endpoint as a safe T_min
        T_min = right
    else:
        T_min = min(roots)

# Sanity check: ensure that for times after T_min, |u| <= 0.1 (sample check)
check_times = np.linspace(T_min, T_min + 60.0, 2000)
max_after = np.max(np.abs(u(check_times)))
print("T_min =", T_min)
print("max |u| on [T_min, T_min+60] =", max_after)
