import numpy as np
import matplotlib.pyplot as plt

# A transformation going from 0 to 1 and 0 to 1 but pushing towards either 0 or 1
def inter01Transformation(base, power01):
    return np.power(base, power01 / (1 - power01))

AVALUE = 50
BVALUE = 50

# Create grid
a_vals = np.linspace(0.001, 0.999, AVALUE)
b_vals = np.linspace(0.001, 0.999, BVALUE)  # avoid division by zero

A, B = np.meshgrid(a_vals, b_vals)

Z = inter01Transformation(A, B)

# Plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(A, B, Z)

ax.set_xlabel("base (a)")
ax.set_ylabel("power01 (b)")
ax.set_zlabel("output")

plt.show()