#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np

def converge(x, y):
    theta = np.arctan2(y,x)
    return -math.cos(theta), -math.sin(theta)

def diverge(x, y):
    theta = np.arctan2(y,x)
    return math.cos(theta), math.sin(theta)

def move_left(x, y):
    return 1.0, 0.0

def move_right(x, y):
    return -1.0, 0.0 

def rotate(x, y):
    theta = np.arctan2(y,x) + np.pi / 2.0
    return -math.cos(theta), -math.sin(theta)

plt.ylim([-21,20])
plt.xlim([-21,20])
for i in np.arange(-20, 20 ,2):
    for j in np.arange(-20, 20, 2):
        if i != 0.0 or j != 0.0:
            x, y = rotate(i, j)
            plt.arrow(i, j, x, y, head_width=0.5, head_length=0.2, fc='k', ec='k')

plt.show()

