import matplotlib.pyplot as plt
import numpy as np
import sys
import os

if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
    raise "Usage\npython3 display.py <filename>"

Z = None
fileName = sys.argv[1].strip()

with open(fileName, 'r') as file:
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
color_bar = ax.contourf(X, Y, Z, levels=24)
ax.set_title('Lid Driven Cavity-Velocity Distribution')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.colorbar(color_bar)
plt.savefig(sys.argv[1].strip().split('.')[-2].strip('/') + '.png')
