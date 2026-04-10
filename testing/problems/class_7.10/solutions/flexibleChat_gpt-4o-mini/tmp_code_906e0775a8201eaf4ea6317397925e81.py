import numpy as np
import matplotlib.pyplot as plt

# Parameters
radius = 1  # Radius of the hemisphere
angle_theta = np.arccos(2/3)  # Angle at which the particle leaves the hemisphere in radians

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Create hemisphere
theta = np.linspace(0, np.pi, 100)

# X and Y coordinates for the hemisphere
x = radius * np.sin(theta)
y = radius * np.cos(theta)

# Plot hemisphere
ax.fill(x, y, color='lightblue', label='Hemisphere')

# Particle trajectory (from top to the angle theta)
particle_x = [0, radius * np.sin(angle_theta)]
particle_y = [radius, radius * np.cos(angle_theta)]
ax.plot(particle_x, particle_y, color='red', linewidth=2, label='Particle Trajectory')

# Draw angle theta
ax.plot([0, radius * np.sin(angle_theta)], [radius, radius * np.cos(angle_theta)], 'k--')
ax.text(radius * np.sin(angle_theta)/2, radius/2, r'$	heta$', fontsize=12, ha='center')

# Set limits and labels
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_aspect('equal')
ax.axhline(0, color='black',linewidth=0.5, ls='--')
ax.axvline(0, color='black',linewidth=0.5, ls='--')
ax.set_title('Particle Trajectory on a Hemisphere')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()

# Show plot
plt.grid()
plt.show()