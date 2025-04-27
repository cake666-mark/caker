import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
f = 50  # 焦距，单位：厘米

# Generate mesh grid for plotting paraboloid
x = np.linspace(-50, 50, 400)
y = np.linspace(-50, 50, 400)
x, y = np.meshgrid(x, y)
z = (x**2 + y**2) / (4 * f)

# Plotting the paraboloid
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

# Setting the plot labels
ax.set_xlabel('X axis (cm)')
ax.set_ylabel('Y axis (cm)')
ax.set_zlabel('Z axis (cm)')
ax.set_title('Paraboloid Surface for Solar Concentration')

# Marking the focus point
focus_point = [0, 0, f]
ax.scatter(*focus_point, color='red', s=100, label='Focus Point')
ax.legend()

plt.show()
