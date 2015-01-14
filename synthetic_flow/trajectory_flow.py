#!/usr/bin/env python

# trajectory_flow.py

# Copyright (C) 2014 Santosh Thoduka

# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.

import math
import numpy as np

class TrajectoryFlow:

    def in_range(self, i, j, dx, dy, xmax, ymax):
        x_max = xmax
        y_max = ymax
        x_min = 0.0
        y_min = 0.0
        if i + dx < x_max and i + dx > x_min and j + dy < y_max and j + dy > y_min:
            return True
        else:
            return False
            
    
    def create_trajectory(self, dx, dy, length=5):
        trajectory = []
        for i in xrange(dx.shape[0]):
            for j in xrange(dx.shape[1]):
                if abs(dx[i,j]) > 0.0 or abs(dy[i,j]) > 0.0:
                    t = []
                    t.append(i)
                    t.append(j)
                    if self.in_range(i,j,dx[i,j],dy[i,j],dx.shape[0],dx.shape[1]):
                        t.append(i+dx[i,j])
                        t.append(j+dy[i,j])
                    trajectory.append(t)            

        for i in xrange(length-2):
            for t in trajectory:
                if self.in_range(t[-2], t[-1], dx[t[0],t[1]], dy[t[0],t[1]], dx.shape[0], dx.shape[1]):
                    t.append(t[-2] + dx[t[0],t[1]])
                    t.append(t[-2] + dy[t[0],t[1]])
        final_traj = []
        for t in trajectory:
            if len(t) == length * 2:
                final_traj.append(t)
        final_traj = np.array(final_traj)
        return final_traj


