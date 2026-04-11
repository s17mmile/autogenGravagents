import math

class TankDrainModel:
    def __init__(self, h0=3.0, R=1.0, r_out=0.1, g=9.81, Cd=1.0, enable_plot=True):
        # Basic validation
        if h0 <= 0:
            raise ValueError("Initial height h0 must be positive.")
        if R <= 0:
            raise ValueError("Tank radius R must be positive.")
        if r_out <= 0:
            raise ValueError("Outlet radius r_out must be positive.")
        if g <= 0:
            raise ValueError("Gravitational acceleration g must be positive.")
        if Cd <= 0:
            raise ValueError("Discharge coefficient Cd must be positive.")

        self.h0 = float(h0)
        self.R = float(R)
        self.r_out = float(r_out)
        self.g = float(g)
        self.Cd = float(Cd)
        self.enable_plot = bool(enable_plot)

        # Areas
        self.A_out = math.pi * self.r_out**2
        self.A_tank = math.pi * self.R**2

        # Analytic rate parameter (alpha) in dh/dt = -alpha * sqrt(h)
        # alpha = (Cd * A_out / A_tank) * sqrt(2 g)
        self.alpha = (self.Cd * self.A_out / self.A_tank) * math.sqrt(2.0 * self.g)

        # Drain time to reach outlet level (h -> 0): t_final = 2*sqrt(h0) / alpha
        self.t_final = 2.0 * math.sqrt(self.h0) / self.alpha

        # Convenience: avoid division by zero in edge cases (already guarded by input checks)

    def h_of_t(self, t):
        # Height as a function of time under ideal Torricelli with Cd: h(t) = (sqrt(h0) - (alpha/2) t)^2, floored at 0
        s = math.sqrt(self.h0) - 0.5 * self.alpha * t
        return max(0.0, s * s)

    def Q_of_t(self, t):
        # Discharge rate at time t: Q = Cd * A_out * sqrt(2 g h)
        h = self.h_of_t(t)
        return self.Cd * self.A_out * math.sqrt(2.0 * self.g * h)

    def V_of_t(self, t):
        # Tank volume above outlet as a function of time: V = A_tank * h(t)
        return self.A_tank * self.h_of_t(t)

    def drain_curve(self, num=200):
        # Generate a time series for plotting or analysis
        if self.t_final <= 0:
            return [0.0], [self.h0]
        times = [i * (self.t_final / (num - 1)) for i in range(num)]
        heights = [self.h_of_t(tt) for tt in times]
        return times, heights

    def save_plot(self, filename="drain_curve.png"):
        # Attempt to save a plot of height vs time; gracefully degrade if plotting is unavailable
        times, heights = self.drain_curve()
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("Plotting not available in this environment. Skipping plot save.")
            return False
        plt.plot(times, heights)
        plt.xlabel("Time (s)")
        plt.ylabel("Height h(t) (m)")
        plt.title("Tank height vs time under Torricelli drain")
        plt.grid(True)
        plt.savefig(filename)
        return True


if __name__ == "__main__":
    # Preset geometry and parameters matching the problem statement
    model = TankDrainModel(h0=3.0, R=1.0, r_out=0.1, g=9.81, Cd=1.0, enable_plot=True)

    print("Drain time to outlet level (ideal Torricelli): {0:.6f} s".format(model.t_final))

    # Sample heights at fractions of the drain time
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = model.t_final * frac
        print("t = {0:.4f} s, h(t) = {1:.6f} m".format(t, model.h_of_t(t)))

    # Optional plot
    if model.enable_plot:
        model.save_plot("drain_curve.png")
