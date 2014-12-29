#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np
from ego_flow import EgomotionFlow
from motion_flow import MotionFlow

height = 240
width = 320
step = 10 
mf = MotionFlow()
ef = EgomotionFlow()
mf.length = 5.0
ef.length = 5.0

plt.ylim([0,height])
plt.xlim([0,width])


dx, dy = ef.get_ccw_rot(width, height, step)
mx, my = mf.get_move_down(width, height, 10, 10, 50, 50, step)

dx = dx + mx
dy = dy + my

for i in xrange(dx.shape[0]):
    for j in xrange(dx.shape[1]):
        if dx[i,j] != 0.0 or dy[i,j] != 0.0:
            plt.arrow(i, j, dx[i,j], dy[i,j], head_width=3.0, head_length=0.8, fc='k', ec='k')

plt.show()
