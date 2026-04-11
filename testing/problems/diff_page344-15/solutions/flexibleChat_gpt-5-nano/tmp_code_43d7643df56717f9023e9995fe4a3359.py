import math

def compute_k_for_peak(y_target: float = 2.0, gamma: float = 0.5):
    """
    Compute the impulse magnitude k required to achieve a peak y_peak = y_target
    for the ODE y'' + gamma y' + y = k delta(t-1), with impulse at t=1.
    Handles underdamped (gamma < 2), critically damped (gamma == 2), and
    overdamped (gamma > 2) regimes analytically.
    Returns (k1, t_peak).
    """
    alpha = gamma / 2.0
    eps = 1e-12
    if gamma < 2.0 - eps:
        # underdamped case: y(t) = (k/omega) e^{-alpha s} sin(omega s), s = t-1
        omega = math.sqrt(max(0.0, 1.0 - alpha*alpha))
        phi = math.atan(omega / alpha)
        s_star = phi / omega
        # peak value for unit impulse
        h_peak = math.exp(-alpha * s_star) * math.sin(omega * s_star) / omega
        k1 = y_target / h_peak
        t_peak = 1.0 + s_star
    elif abs(gamma - 2.0) <= eps:
        # critical damping: y(t) = k s e^{-s}, s = t-1
        t_peak = 2.0
        h_peak = 1.0 / math.e
        k1 = y_target / h_peak
    else:
        # overdamped: y(t) = A (e^{r1 s} - e^{r2 s}), s = t-1
        a = gamma / 2.0
        d = math.sqrt(max(0.0, a*a - 1.0))
        r1 = -a + d
        r2 = -a - d
        s_star = math.log(r2 / r1) / (r1 - r2)
        h_peak = - math.exp(r1 * s_star) / r2
        k1 = y_target / h_peak
        t_peak = 1.0 + s_star
    return k1, t_peak

k1, t_peak = compute_k_for_peak()
print("k1 = {:.6f}".format(k1))
print("t_peak = {:.6f}".format(t_peak))