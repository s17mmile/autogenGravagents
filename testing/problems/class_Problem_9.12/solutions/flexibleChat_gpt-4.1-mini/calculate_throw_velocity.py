# filename: calculate_throw_velocity.py
import math

def calculate_throw_velocity(M_initial=100.0, m_gas=2.0, m_tank_total=10.0, v_exhaust=100.0, v_orbiter=3.0):
    """
    Calculate the velocity at which Stumblebum must throw the empty tank backward
    to reach the orbiter moving away at a given velocity.

    Parameters:
    - M_initial: Total initial mass including gas and tank (kg)
    - m_gas: Mass of gas expelled (kg)
    - m_tank_total: Total mass of the tank including gas (kg)
    - v_exhaust: Velocity of gas relative to Stumblebum (m/s)
    - v_orbiter: Velocity of orbiter away from Stumblebum (m/s)

    Returns:
    - v_throw: Velocity at which the empty tank must be thrown backward relative to Stumblebum (m/s)
    """
    m_tank = m_tank_total - m_gas  # Mass of empty tank

    # Step 1: Calculate velocity after expelling gas using rocket equation
    M_final = M_initial - m_gas
    v_after_gas = v_exhaust * math.log(M_initial / M_final)

    # Step 2: Calculate velocity to throw empty tank to reach orbiter
    # Conservation of momentum:
    # (M_final) * v_after_gas = (M_final - m_tank) * v_s + m_tank * (v_s - v_throw)
    # We want v_s = v_orbiter
    # Solve for v_throw:
    numerator = (M_final * v_after_gas) - ((M_final - m_tank) * v_orbiter)
    v_throw = numerator / m_tank

    return v_throw

if __name__ == "__main__":
    throw_velocity = calculate_throw_velocity()
    print(f"Stumblebum must throw the empty tank backward at {throw_velocity:.2f} m/s relative to himself to reach the orbiter.")
