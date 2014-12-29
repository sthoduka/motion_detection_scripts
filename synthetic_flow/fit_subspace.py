#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_fit_error(data, num_motions):
    origdata = data.copy()
    data = data.T

    xsum = np.sum(data[0,:])
    ysum = np.sum(data[1,:])
    xsum /= data.shape[1]
    ysum /= data.shape[1]
    data[::2,:] -= xsum
    data[1::2,:] -= ysum

    num_frames = data.shape[0] / 2
    n = 2*num_frames
    m = num_motions
    num_max = 0
    proj = np.empty((n,n))
    proj_residual = np.empty((data.shape[1],1))
    print "data: " , data.shape
    for i in xrange(100):
        num = 0
        d = 4*m
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
            if r < (n-d)*(0.05**2):
                num = num + 1

        if num > num_max:       
            num_max = num
            index = rand_indices.copy()
            proj = Pnd.copy()
            proj_residual = residual.copy()
    
    return proj_residual, index
