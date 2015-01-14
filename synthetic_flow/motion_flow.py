#!/usr/bin/env python

# motion_flow.py

# Copyright (C) 2014 Santosh Thoduka

# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.

import math
import matplotlib.pyplot as plt
import numpy as np

class MotionFlow:
    def __init__(self):
        self.length = 1.0
    def move_left(self, x, y):
        return -self.length, 0.0

    def move_right(self, x, y):
        return self.length, 0.0 

    def move_down(self, x, y):
        return 0.0, -self.length

    def move_up(self, x, y):
        return 0.0, self.length 

    def get_move_left(self, x_size, y_size, xpos, ypos, width,height, step):
        dx = np.zeros((x_size, y_size))
        dy = np.zeros((x_size, y_size))
        for i in np.arange(xpos, xpos+width, step):
            for j in np.arange(ypos, ypos+height, step):
                xdiff, ydiff = self.move_left(i, j)
                dx[i,j] = xdiff
                dy[i,j] = ydiff
        return dx, dy

    def get_move_right(self, x_size, y_size, xpos, ypos, width,height, step):
        dx = np.zeros((x_size, y_size))
        dy = np.zeros((x_size, y_size))
        for i in np.arange(xpos, xpos+width, step):
            for j in np.arange(ypos, ypos+height, step):
                xdiff, ydiff = self.move_right(i, j)
                dx[i,j] = xdiff
                dy[i,j] = ydiff
        return dx, dy

    def get_move_up(self, x_size, y_size, xpos, ypos, width,height, step):
        dx = np.zeros((x_size, y_size))
        dy = np.zeros((x_size, y_size))
        for i in np.arange(xpos, xpos+width, step):
            for j in np.arange(ypos, ypos+height, step):
                xdiff, ydiff = self.move_up(i, j)
                dx[i,j] = xdiff
                dy[i,j] = ydiff
        return dx, dy

    def get_move_down(self, x_size, y_size, xpos, ypos, width,height, step):
        dx = np.zeros((x_size, y_size))
        dy = np.zeros((x_size, y_size))
        for i in np.arange(xpos, xpos+width, step):
            for j in np.arange(ypos, ypos+height, step):
                xdiff, ydiff = self.move_down(i, j)
                dx[i,j] = xdiff
                dy[i,j] = ydiff
        return dx, dy
