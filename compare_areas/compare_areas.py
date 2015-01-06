#!usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

class CompareAreas:
    # frame_number, left, top, right, bottom
    def read_kitti_file(self, filename):
        data = np.genfromtxt(filename, dtype=np.float64, delimiter=' ', usecols=(0,6,7,8,9))
        object_type = np.genfromtxt(filename, dtype=None, delimiter=' ', usecols=(2))
        new_data = data[np.where(object_type != "DontCare")]
        return new_data.astype(int)
    
    def read_motion_detection_file(self, filename):
        data = np.genfromtxt(filename, dtype=None, delimiter=',', usecols=(0,2,3,4,5))
        return data

    def construct_mask(self, width, height, frame_number, data, start_index):
        mask = np.zeros((height, width), dtype=bool)
        current_index = start_index      
        while True:
            if current_index >= data.shape[0] or data[current_index,0] != frame_number:
                break
            else:
                col_start = data[current_index,1]
                col_stop = data[current_index,3]
                row_start = data[current_index,2]
                row_stop = data[current_index,4]
                mask[row_start:row_stop, col_start:col_stop] = True
            current_index += 1
        return mask, current_index

    def get_confusion_matrix(self, gt_mask, md_mask):
        tp = np.sum(np.bitwise_and(gt_mask, md_mask))
        fp = np.sum(np.bitwise_and(md_mask, np.invert(gt_mask)))
        fn = np.sum(np.bitwise_and(np.invert(md_mask), gt_mask))
        tn = np.sum(np.bitwise_and(np.invert(gt_mask), np.invert(md_mask)))
        accuracy = (tp + tn) / float(tp + fp + fn + tn)
        if tp + fn == 0.0:
            tpr = 0.0
        else:
            tpr = tp / float(tp + fn)
        if tp + fp == 0.0:
            precision = 0.0
        else:
            precision = tp / float(tp + fp)
        return [tpr, precision, accuracy]

    def get_area_match_list(self, gt_file, md_file, width, height):
        gt = self.read_kitti_file(gt_file)
        md = self.read_motion_detection_file(md_file)
        frame_count = gt[-1,0]
        gt_start_index = 0
        md_start_index = 0
        frame_number = 0
        match_list = []
        while frame_number <= frame_count:
            md_mask, md_start_index = self.construct_mask(width, height, frame_number, md, md_start_index)
            gt_mask, gt_start_index = self.construct_mask(width, height, frame_number, gt, gt_start_index)
            match = self.get_confusion_matrix(gt_mask, md_mask)
            match_list.append(match)
            frame_number += 1
        match_list = np.array(match_list)
        return match_list

rd = CompareAreas()
match_list = rd.get_area_match_list("motion_gt/0000_motion.txt", "motion_md/0000.log", 1242, 375)

tpr, = plt.plot(match_list[:,0], label="TPR")
precision, = plt.plot(match_list[:,1], label="Precision")
acc, = plt.plot(match_list[:,2], label="Accuracy")
plt.legend([tpr, precision, acc], ["TPR", "Precision", "Accuracy"])
plt.title("Sequence 0000")
plt.xlabel("Frame number")
plt.show(block=False)

plt.figure()

match_list = rd.get_area_match_list("motion_gt/0001_motion.txt", "motion_md/0001.log", 1242, 375)
tpr, = plt.plot(match_list[:,0], label="TPR")
precision, = plt.plot(match_list[:,1], label="Precision")
acc, = plt.plot(match_list[:,2], label="Accuracy")
plt.legend([tpr, precision, acc], ["TPR", "Precision", "Accuracy"])
plt.title("Sequence 0001")
plt.xlabel("Frame number")
plt.show(block=False)

plt.figure()

match_list = rd.get_area_match_list("motion_gt/0002_motion.txt", "motion_md/0002.log", 1242, 375)
tpr, = plt.plot(match_list[:,0], label="TPR")
precision, = plt.plot(match_list[:,1], label="Precision")
acc, = plt.plot(match_list[:,2], label="Accuracy")
plt.legend([tpr, precision, acc], ["TPR", "Precision", "Accuracy"])
plt.title("Sequence 0002")
plt.xlabel("Frame number")
plt.show(block=False)

plt.figure()

match_list = rd.get_area_match_list("motion_gt/0003_motion.txt", "motion_md/0003.log", 1242, 375)
tpr, = plt.plot(match_list[:,0], label="TPR")
precision, = plt.plot(match_list[:,1], label="Precision")
acc, = plt.plot(match_list[:,2], label="Accuracy")
plt.legend([tpr, precision, acc], ["TPR", "Precision", "Accuracy"])
plt.title("Sequence 0003")
plt.xlabel("Frame number")
plt.show(block=True)
