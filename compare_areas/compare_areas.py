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
            ppv = 0.0
        else:
            ppv = tp / float(tp + fp)
        if ppv != 0.0:
            fdr = 1.0 - ppv
        else:
            fdr = 0.0
        #return [np.sum(tp), np.sum(fp), np.sum(fn), np.sum(tn)]
        return [accuracy, tpr ,ppv, fdr]

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
#match_list = rd.get_area_match_list("0012.txt", "0012.log", 1242, 375)
match_list = rd.get_area_match_list("0015.txt", "0015.log", 1224, 370)
#match_list = rd.get_area_match_list("0000.txt", "0000_md.log", 1242, 375)
acc, = plt.plot(match_list[:,0], label="Accuracy")
tpr, = plt.plot(match_list[:,1], label="TPR")
ppv, = plt.plot(match_list[:,2], label="PPV")
fdr, = plt.plot(match_list[:,3], label="FDR")
plt.legend([acc, tpr, ppv, fdr], ["Accuracy", "TPR", "PPV", "FDR"])
plt.show()
