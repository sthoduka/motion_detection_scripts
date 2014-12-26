import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pylab
import numpy as np
import glob
import cv2
import matplotlib.cm as cm

from outlier.py import is_outlier

import re

# the following two functions were taken from here:
# http://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def cv2numpy(cvarr, the_type):
  a = np.asarray(cvarr, dtype=the_type)
  return a



f = glob.glob("lots_of_movement/*")

f.sort(key=natural_keys)
files = np.array(f)
cap = cv2.VideoCapture("lots_of_movement.avi")

v_data = None
h_data = None
plt.figure(1)
plt.ylim([0,240])
plt.xlim([0,320])
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

plt.ion()
frame_num = 0
stationary = False 
for f in files:
    if f.endswith("_f"):
        v_data = np.genfromtxt(f, dtype=np.float64, delimiter=',')
        angles = []
        lengths = []
        plt.figure(1)
        plt.cla()
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        frame_num = frame_num + 1

    if f.endswith("_h"):
        print f
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_ = cv2numpy(gray, 'uint8')
        h_data = np.genfromtxt(f, dtype=np.float64, delimiter=',')
        for y, row in enumerate(h_data):
            for x, col in enumerate(row):
                if stationary:
                    angle = np.arctan2(v_data[y][x], col)
                    angles.append(angle)
                    lengths.append(np.sqrt(v_data[y][x]**2 + col**2))
                else:
                    if abs(col) > 0.0 or abs(v_data[y][x]) > 0.0:
                        angle = np.arctan2(v_data[y][x], col)
                        angles.append(angle)
                        lengths.append(np.sqrt(v_data[y][x]**2 + col**2))
        index = 0
        angles = np.array(angles)
        angle_outlier = is_outlier(angles, frame_num == 65,3.5)
        lengths = np.array(lengths)
        length_outlier = is_outlier(lengths, frame_num == 65, 3.5)

        plt.figure(1)
        plt.title("Frame %i" %frame_num)
        plt.imshow(image_, cmap=cm.gray)
        for y, row in enumerate(h_data):
            for x, col in enumerate(row):
                if abs(col) > 0.0 or abs(v_data[y][x]) > 0.0:                    
                    plot_x = x * 10;
                    plot_y = y * 10;
                    if angle_outlier[index] or length_outlier[index]:
                        #print "outlier at: ", y, x
                        plt.arrow(plot_x, plot_y, col, v_data[y][x], head_width=2, head_length=2, fc='r', ec='r')
                    else:
                        plt.arrow(plot_x, plot_y, col, v_data[y][x], head_width=2, head_length=2, fc='g', ec='g')
                    if stationary == False:
                        index = index + 1                
                if stationary:
                    index = index + 1



        plt.figure(2)
        filtered_angles = angles[~angle_outlier]
        filtered_lengths = lengths[~length_outlier]
        ax1.set_title("Angles")
        ax1.set_xlabel("radian")
        ax1.hist(angles, bins=np.arange(-np.pi,np.pi,np.pi/10.0))
        #plt.figure(3)
        #plt.title("Angles Filtered")
        ax2.set_title("Filtered Angles")
        ax2.set_xlabel("radian")
        ax2.hist(filtered_angles, bins=np.arange(-np.pi,np.pi,np.pi/10.0))
        ax3.set_title("Lengths")
        ax3.set_xlabel("pixels")
        ax3.hist(lengths, bins=np.arange(0, 20, 1))
        ax4.set_title("Filtered Lengths")
        ax4.set_xlabel("pixels")
        ax4.hist(filtered_lengths, bins=np.arange(0, 20, 1))
        plt.draw()
plt.show(block=True)
