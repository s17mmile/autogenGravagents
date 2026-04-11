import math


def cone_volume_and_errors(r: float, h: float, dr: float, dh: float) -> dict:
    """Compute cone volume and error bounds given measurements r, h with perturbations dr, dh.

    Returns a dict with keys: V, dV_max, dV_rms, V_min, V_max, relative_max_error.
    - V: cone volume
    - dV_max: conservative, sign-agnostic maximum change in V due to dr and dh
    - dV_rms: root-sum-square bound for independent perturbations
    - V_min, V_max: volume interval
    - relative_max_error: dV_max / V (or inf if V == 0)
    """
    if r <= 0 or h <= 0:
        raise ValueError("r and h must be positive to have a valid cone volume.")

    V = (1.0/3.0) * math.pi * r * r * h
    dV_dr = (2.0/3.0) * math.pi * r * h
    dV_dh = (1.0/3.0) * math.pi * r * r

    # Conservative, sign-agnostic bound for maximum volume change
    dV_max = abs(dV_dr) * abs(dr) + abs(dV_dh) * abs(dh)

    # RMS bound for independent errors
    dV_rms = math.sqrt((dV_dr * dr) ** 2 + (dV_dh * dh) ** 2)

    V_min = V - dV_max
    V_max = V + dV_max
    relative_max_error = dV_max / V if V != 0 else float('inf')

    return {
        'V': V,
        'dV_max': dV_max,
        'dV_rms': dV_rms,
        'V_min': V_min,
        'V_max': V_max,
        'relative_max_error': relative_max_error
    }


def main():
    r = 10.0
    h = 25.0
    dr = 0.1
    dh = 0.1

    res = cone_volume_and_errors(r, h, dr, dh)
    print('V = {:.6f} cm3'.format(res['V']))
    print('dV_max = {:.6f} cm3'.format(res['dV_max']))
    print('dV_rms = {:.6f} cm3'.format(res['dV_rms']))
    print('V range: [{:.6f}, {:.6f}] cm3'.format(res['V_min'], res['V_max']))
    if res['relative_max_error'] == float('inf'):
        print('Relative max error: infinite (V is zero).')
    else:
        print('Relative max error = {:.6%}'.format(res['relative_max_error']))


if __name__ == '__main__':
    main()