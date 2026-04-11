import math

# Constants (SI)
c = 299_792_458.0
L = 100.0          # ground-frame separation between markers (m)
dt = 0.4e-6        # 0.4 microseconds
v = L / dt           # racer speed (m/s)
beta = v / c
gamma = 1.0 / math.sqrt(1.0 - beta**2)
L_prime = L / gamma    # contracted separation in racer frame

print("Racer speed v = {0:.6g} m/s".format(v))
print("beta = v/c = {0:.6f}".format(beta))
print("gamma = {0:.6f}".format(gamma))
print("Length contracted separation L' in racer frame = {0:.6f} m".format(L_prime))


def apparent_separation_symmetric(L: float, h: float, D: float, beta: float) -> float:
    """Approximate Terrell-Penrose-inspired optical separation for a symmetric setup.

    Geometry (ground frame): markers at A = (-L/2, 0, 0), B = (+L/2, 0, 0).
    Eye (observer) at (0, 0, h). The observer moves with velocity beta*c along +x.

    This function uses standard aberration formulas (cos' = (cos - beta) / (1 - beta cos),
    sin' = sin / (gamma * (1 - beta cos))) to estimate the apparent separation on a screen
    at distance D along the line of sight. This is a simplified proxy, not a full retarded-time
    Terrell-Penrose calculation.

    Returns the apparent lateral separation on the screen (in meters).
    """
    c = 299_792_458.0  # speed of light, not strictly needed here but kept for clarity
    gamma_local = 1.0 / math.sqrt(1.0 - beta**2)

    Ex, Ey, Ez = 0.0, 0.0, h
    Ax, Ay, Az = -L/2.0, 0.0, 0.0
    Bx, By, Bz =  L/2.0, 0.0, 0.0

    # Vectors eye -> A and eye -> B (in ground frame): dx, dz components
    dxA = Ax - Ex
    dzA = Az - Ez  # negative since eye is above ground
    distA = math.hypot(dxA, dzA)
    cosA = dxA / distA
    sinA = dzA / distA

    dxB = Bx - Ex
    dzB = Bz - Ez
    distB = math.hypot(dxB, dzB)
    cosB = dxB / distB
    sinB = dzB / distB

    # Aberration transforms to moving (eye) frame
    cosA_p = (cosA - beta) / (1.0 - beta * cosA)
    cosB_p = (cosB - beta) / (1.0 - beta * cosB)
    sinA_p = sinA / (gamma_local * (1.0 - beta * cosA))
    sinB_p = sinB / (gamma_local * (1.0 - beta * cosB))

    # Numerical safety
    cosA_p = max(-1.0, min(1.0, cosA_p))
    cosB_p = max(-1.0, min(1.0, cosB_p))
    eps = 1e-12
    if abs(cosA_p) < eps:
        cosA_p = eps if cosA_p >= 0 else -eps
    if abs(cosB_p) < eps:
        cosB_p = eps if cosB_p >= 0 else -eps

    tanA_p = sinA_p / cosA_p
    tanB_p = sinB_p / cosB_p

    separation = D * (tanA_p - tanB_p)
    return separation

# Test: print apparent separations for several eye heights
heights = [0.5, 1.0, 2.0, 5.0]
D = 100.0  # screen distance along line of sight (m)
for h in heights:
    sep = apparent_separation_symmetric(L, h, D, beta)
    print("h = {0:.2f} m: apparent separation ~ {1:.6f} m".format(h, sep))

# End of snippet