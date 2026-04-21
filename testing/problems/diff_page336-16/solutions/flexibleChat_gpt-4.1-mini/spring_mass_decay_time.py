# filename: spring_mass_decay_time.py
import numpy as np
import math

omega = math.sqrt(63)/8


def homogeneous_solution(t, A, B):
    """
    Compute the homogeneous solution u_h(t) = e^{-t/8} (A cos(omega t) + B sin(omega t)).

    Parameters:
    t : float or np.array
        Time variable.
    A, B : float
        Coefficients determined by initial conditions.

    Returns:
    float or np.array
        Value of the homogeneous solution at time t.
    """
    return np.exp(-t/8)*(A*np.cos(omega*t) + B*np.sin(omega*t))


def homogeneous_solution_derivative(t, A, B):
    """
    Compute the derivative of the homogeneous solution u_h'(t).

    Parameters:
    t : float or np.array
        Time variable.
    A, B : float
        Coefficients determined by initial conditions.

    Returns:
    float or np.array
        Value of the derivative of the homogeneous solution at time t.
    """
    exp_term = np.exp(-t/8)
    cos_term = np.cos(omega*t)
    sin_term = np.sin(omega*t)
    return exp_term*(-(1/8)*(A*cos_term + B*sin_term) + (-A*omega*sin_term + B*omega*cos_term))


def compute_tau(k=2, threshold=0.1):
    """
    Compute the time tau after which |u(t)| < threshold for all t > tau.

    Parameters:
    k : float
        Parameter multiplying the forcing function (default 2).
    threshold : float
        Threshold for |u(t)| (default 0.1).

    Returns:
    float
        Time tau after which |u(t)| < threshold.
    """
    # Step 4: Initial conditions for homogeneous solution on [1.5, 2.5)
    A = -k  # since u_p = k during forcing
    B = -k / math.sqrt(63)

    # Evaluate homogeneous solution and derivative at t=1 (corresponds to t=2.5)
    t1 = 1.0
    u_h_t1 = homogeneous_solution(t1, A, B)
    u_h_prime_t1 = homogeneous_solution_derivative(t1, A, B)

    # Particular solution during forcing is u_p = k
    u_2_5 = k + u_h_t1
    u_prime_2_5 = u_h_prime_t1

    # Step 5: Find coefficients C and D for t >= 2.5
    # General solution: u(t) = e^{-(t-2.5)/8} (C cos(omega (t-2.5)) + D sin(omega (t-2.5)))
    # At t=2.5 (tau=0), u(0) = C, u'(0) = -C/8 + D*omega

    C = u_2_5
    D = (u_prime_2_5 + C/8)/omega

    # Amplitude of oscillation envelope at t=2.5
    M = math.sqrt(C**2 + D**2)

    if M <= threshold:
        # Already below threshold at t=2.5
        tau = 2.5
    else:
        # Solve for tau: M * exp(-(tau - 2.5)/8) = threshold
        # => (tau - 2.5)/8 = -ln(threshold / M)
        tau = 2.5 + 8 * (-math.log(threshold / M))

    return tau


if __name__ == '__main__':
    tau_value = compute_tau(k=2, threshold=0.1)
    print(f"The time tau after which |u(t)| < 0.1 for all t > tau is approximately {tau_value:.4f}.")
