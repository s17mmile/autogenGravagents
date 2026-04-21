# filename: beta_cdf_evaluation.py

try:
    import scipy.special
except ImportError:
    raise ImportError("scipy is required to run this code. Please install it using 'pip install scipy'.")


def beta_cdf(alpha, beta, x):
    """
    Compute the cumulative distribution function (CDF) of the Beta distribution
    with parameters alpha and beta at point x.

    The integral \int_0^x (Gamma(alpha+beta) / (Gamma(alpha) Gamma(beta))) * y^{alpha-1} * (1-y)^{beta-1} dy
    is equal to the regularized incomplete Beta function betainc(alpha, beta, x).
    """
    return scipy.special.betainc(alpha, beta, x)


if __name__ == "__main__":
    alpha = 4
    beta = 3
    x = 0.4
    result = beta_cdf(alpha, beta, x)
    print(f"The value of the integral from 0 to {x} is: {result}")
