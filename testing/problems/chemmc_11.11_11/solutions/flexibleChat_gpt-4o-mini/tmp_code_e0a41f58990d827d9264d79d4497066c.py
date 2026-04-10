import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Define energy levels
energy_levels = [2, 1]  # π1 (bonding) and π2* (antibonding)
labels = ['π1 (bonding)', 'π2* (antibonding)']

# Create a bar plot for the energy levels
ax.barh(labels, energy_levels, color=['blue', 'red'], edgecolor='black')

# Add electrons to the bonding orbital
ax.text(1.5, 0, '2 electrons', ha='center', va='center', fontsize=12, color='white')

# Set the limits and labels
ax.set_xlim(0, 3)
ax.set_xlabel('Energy')
ax.set_title('Molecular Orbital Diagram for Ethene')

# Set y-axis label
ax.set_ylabel('Molecular Orbitals')

# Add grid lines for better visualization
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()