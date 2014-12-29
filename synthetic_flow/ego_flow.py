#!/usr/bin/env python

import math
import numpy as np
import random

class EgomotionFlow:
    def __init__(self):
        self.length = 1.0

    def converge(self, x, y):
        theta = np.arctan2(y,x)
        return -self.length * math.cos(theta), -self.length * math.sin(theta)

    def diverge(self, x, y):
        theta = np.arctan2(y,x)
        return self.length * math.cos(theta), self.length * math.sin(theta)

    def move_left(self, x, y):
        return self.length, 0.0

    def move_right(self, x, y):
        return self.length*(random.random() - 0.5)-self.length, 0.1*self.length*(random.random() - 0.5)
        #return -self.length, 0.0

    def rotate_ccw(self, x, y):
        theta = np.arctan2(y,x) + np.pi / 2.0
        return -self.length * math.cos(theta), -self.length * math.sin(theta)

    def rotate_cw(self, x, y):
        theta = np.arctan2(y,x) + np.pi / 2.0
        return self.length * math.cos(theta), self.length * math.sin(theta)
    
    def get_forward_trans(self, x_size, y_size, step):
        x_lim = x_size / 2
        y_lim = y_size / 2
        dx = np.zeros((x_size, y_size))
        dy = np.zeros((x_size, y_size))
        for i in np.arange(-x_lim, x_lim, step):
            for j in np.arange(-y_lim, y_lim, step):
                if i != 0.0 or j != 0.0:
                    xdiff, ydiff = self.diverge(i, j);
                    dx[i + x_lim, j + y_lim] = xdiff
                    dy[i + x_lim, j + y_lim] = ydiff
        return dx, dy         

    def get_backward_trans(self, x_size, y_size, step):
        x_lim = x_size / 2
        y_lim = y_size / 2
        dx = np.zeros((x_size, y_size))
        dy = np.zeros((x_size, y_size))
        for i in np.arange(-x_lim, x_lim, step):
            for j in np.arange(-y_lim, y_lim, step):
                if i != 0.0 or j != 0.0:
                    xdiff, ydiff = self.converge(i, j);
                    dx[i + x_lim, j + y_lim] = xdiff
                    dy[i + x_lim, j + y_lim] = ydiff
        return dx, dy         

    def get_left_trans(self, x_size, y_size, step):
        x_lim = x_size / 2
        y_lim = y_size / 2
        dx = np.zeros((x_size, y_size))
        dy = np.zeros((x_size, y_size))
        for i in np.arange(-x_lim, x_lim, step):
            for j in np.arange(-y_lim, y_lim, step):
                xdiff, ydiff = self.move_left(i, j);
                dx[i + x_lim, j + y_lim] = xdiff
                dy[i + x_lim, j + y_lim] = ydiff
        return dx, dy         

    def get_right_trans(self, x_size, y_size, step):
        x_lim = x_size / 2
        y_lim = y_size / 2
        dx = np.zeros((x_size, y_size))
        dy = np.zeros((x_size, y_size))
        for i in np.arange(-x_lim, x_lim, step):
            for j in np.arange(-y_lim, y_lim, step):
                xdiff, ydiff = self.move_right(i, j);
                dx[i + x_lim, j + y_lim] = xdiff
                dy[i + x_lim, j + y_lim] = ydiff
        return dx, dy

    def get_cw_rot(self, x_size, y_size, step):
        x_lim = x_size / 2
        y_lim = y_size / 2
        dx = np.zeros((x_size, y_size))
        dy = np.zeros((x_size, y_size))
        for i in np.arange(-x_lim, x_lim, step):
            for j in np.arange(-y_lim, y_lim, step):
                if i != 0.0 or j != 0.0:
                    xdiff, ydiff = self.rotate_cw(i, j);
                    dx[i + x_lim, j + y_lim] = xdiff
                    dy[i + x_lim, j + y_lim] = ydiff
        return dx, dy         

    def get_ccw_rot(self, x_size, y_size, step):
        x_lim = x_size / 2
        y_lim = y_size / 2
        dx = np.zeros((x_size, y_size))
        dy = np.zeros((x_size, y_size))
        for i in np.arange(-x_lim, x_lim, step):
            for j in np.arange(-y_lim, y_lim, step):
                if i != 0.0 or j != 0.0:
                    xdiff, ydiff = self.rotate_ccw(i, j);
                    dx[i + x_lim, j + y_lim] = xdiff
                    dy[i + x_lim, j + y_lim] = ydiff
        return dx, dy         
