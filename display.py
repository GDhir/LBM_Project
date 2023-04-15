import matplotlib.pyplot as plt
import numpy as np
import sys
import os

if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
    raise "Usage\npython3 display.py <filename>"

Z = None

with open(sys.argv[1], 'r') as file:
    line = file.readline()
    NX, NY = map(int, line.strip().split(' '))
    Z = np.zeros((NX, NY))
    for row, line in enumerate(file):
        # print(row, map(float, line.strip().split(' '))
        for y, val in enumerate(map(float, line.strip().split(' '))):
            Z[row, y] = val

Z = Z.T

[X, Y] = np.meshgrid(np.arange(0, Z.shape[0]), np.arange(0, Z.shape[1]))

# plots filled contour plot
fig, ax = plt.subplots(1, 1)
ax.contourf(X, Y, Z)
ax.set_title('Filled Contour Plot')
ax.set_xlabel('feature_x')
ax.set_ylabel('feature_y')
plt.show()
