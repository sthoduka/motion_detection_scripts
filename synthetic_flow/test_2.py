#!/usr/bin/env python

# test_2.py

# Copyright (C) 2014 Santosh Thoduka

# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.

import math
import matplotlib.pyplot as plt
import numpy as np
from ego_flow import EgomotionFlow
from motion_flow import MotionFlow
from trajectory_flow import TrajectoryFlow
from fit_subspace import get_fit_error

height = 240
width = 320
step = 10 

mf = MotionFlow()
ef = EgomotionFlow()
mf.length = 4.0
ef.length = 1.0 

plt.ylim([0,height])
plt.xlim([0,width])


dx, dy = ef.get_cw_rot(width, height, step)
mx, my = mf.get_move_right(width, height, 100, 100, 50, 50, step)
nx, ny = mf.get_move_left(width, height, 210, 200, 20, 20, step)

dx = dx + mx + nx
dy = dy + my + ny


for i in xrange(dx.shape[0]):
    for j in xrange(dx.shape[1]):
        if dx[i,j] != 0.0 or dy[i,j] != 0.0:
            plt.arrow(i, j, dx[i,j], dy[i,j], head_width=3.0, head_length=0.8, fc='k', ec='k')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simulated optical flow vectors")
plt.show(block=False)

plt.figure()
tf = TrajectoryFlow()
traj = tf.create_trajectory(dx,dy,10)
np.savetxt('test.out', traj, fmt='%.4f', delimiter=',')
traj = np.loadtxt("test.out", dtype=np.float64, delimiter=',')
trajcopy = traj.copy()
residuals, index = get_fit_error(trajcopy, 2)
plt.plot(residuals)
plt.xlabel("Trajectory index")
plt.ylabel("Residual")
plt.title("Subspace fit residual")
plt.show(block=False)



plt.figure()
plt.ylim([0,height])
plt.xlim([0,width])
#plt.gca().set_color_cycle(['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666'])
plt.gca().set_color_cycle(['k', 'k', 'k', 'k', 'k', 'k', 'k','k'])
for idx, t in enumerate(traj):
    if idx in index:
        x = t[::2]
        y = t[1::2]
        d = np.vstack((x,y))
        d =  d.T
        plt.plot(d[:,0], d[:,1],linewidth=2.0)
plt.title("Trajectories used to define the subspace")
plt.xlabel("X")
plt.ylabel("Y")
plt.show(block=False)

plt.figure()
plt.ylim([0,height])
plt.xlim([0,width])

plt.gca().set_color_cycle(['k', 'k', 'k', 'k', 'k', 'k', 'k','k'])
for idx, t in enumerate(traj):
    x = t[::2]
    y = t[1::2]
    d = np.vstack((x,y))
    d =  d.T
    plt.plot(d[:,0], d[:,1],linewidth=2.0)
plt.title("All trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.show(block=False)

plt.figure()
plt.ylim([0,height])
plt.xlim([0,width])

for idx,r in enumerate(residuals):
    if r > 0.0000002:
        plt.scatter(traj[idx,-4], traj[idx,-3])

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Outlier points")
plt.show()
