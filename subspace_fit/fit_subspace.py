#!/usr/bin/env python

'''
fit_subspace.py

Copyright (C) 2014 Santosh Thoduka

This software may be modified and distributed under the terms
of the MIT license.  See the LICENSE file for details.
'''



import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm


frame = cv2.imread("frame.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def cv2numpy(cvarr, the_type):
  a = np.asarray(cvarr, dtype=the_type)
  return a

image_ = cv2numpy(gray, 'uint8')

data = np.genfromtxt("frame1", dtype=np.float64, delimiter=',')
origdata = np.genfromtxt("frame1", dtype=np.float64, delimiter=',')
data = data.T
xsum = np.sum(data[0,:])
ysum = np.sum(data[1,:])
xsum /= data.shape[1]
ysum /= data.shape[1]
data[::2,:] -= xsum
data[1::2,:] = ysum - data[1::2,:]

num_frames = 10
n = 2*num_frames
m = 2
num_max = 0
proj = np.empty((n,n))
proj_residual = np.empty((data.shape[1],1))
for i in xrange(50):
    num = 0
    d = 8*m
    rand_indices = np.random.randint(data.shape[1], size=d)

    subset = data[:,rand_indices]

    U,s,Vt = np.linalg.svd(subset)

    Pnd = np.zeros((n,n))
    for idx in xrange(d):
        M = U[:,idx][:,np.newaxis].dot(U[:,idx][:,np.newaxis].T)
        Pnd = Pnd + M
    Pnd = np.eye(n) - Pnd

    residual = data.T.dot(Pnd.dot(data)).diagonal()[:,np.newaxis]

    for r in residual:
        if r < (n-d)*(0.5**2):
            num = num + 1

    if num > num_max:
        num_max = num
        proj = Pnd.copy()
        proj_residual = residual.copy()
    print i

plt.plot(proj_residual)
plt.show(block=False)
plt.figure()
plt.imshow(image_, cmap=cm.gray)
for idx,r in enumerate(proj_residual):
    if r > 0.2:
        plt.scatter(origdata[idx,-4], 240-origdata[idx,-3])
plt.show(block=True)
