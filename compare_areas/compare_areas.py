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

    # frame number, object id, left, top, right, bottom
    def read_kitti_file_with_id(self, filename):
        data = np.genfromtxt(filename, dtype=np.float64, delimiter=' ', usecols=(0,1,6,7,8,9))
        return data.astype(int)

    # frame number, left, top, right, bottom
    def read_motion_detection_file(self, filename):
        data = np.genfromtxt(filename, dtype=None, delimiter=',', usecols=(0,2,3,4,5))
        return data
    
    # frame number, object id, left, top, right, bottom
    def read_motion_detection_file_with_id(self, filename):
        data = np.genfromtxt(filename, dtype=None, delimiter=',', usecols=(0,1,2,3,4,5))
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

    def construct_single_mask(self, width, height, corners):
        mask = np.zeros((height, width), dtype=bool)
        col_start = corners[0]
        col_stop = corners[2]
        row_start = corners[1]
        row_stop = corners[3]
        mask[row_start:row_stop, col_start:col_stop] = True
        return mask
        

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

    def get_detection_rate(self, gt_file, md_file, width, height):
        gt = self.read_kitti_file_with_id(gt_file)
        md = self.read_motion_detection_file_with_id(md_file)
        frame_count = gt[-1,0]

        md_index = 0
        md_index_start = 0
        total_detected = 0
        objects = dict()
        detections = dict()

        for row in gt:
            frame_gt = row[0]
            if frame_gt in objects:
                objects[frame_gt] += 1
            else:
                objects[frame_gt] = 1
            if frame_gt not in detections:
                detections[frame_gt] = 0


            obj_id_git = row[1]           
            detected = False 
            no_detection = False 
            md_index = md_index_start
            if md_index >= md.shape[0]:
                break

            while True:
                frame_md = md[md_index,0]
                if frame_md == frame_gt:
                    md_index_start = md_index
                    break
                elif frame_md > frame_gt:
                    no_detection = True
                    break
                else:
                    md_index += 1
                    if md_index >= md.shape[0]:
                        no_detection = True
                        md_index_start = md_index
                        break

            if no_detection:
                continue
            while True:
                frame_md = md[md_index,0]
                if frame_md != frame_gt:
                    break
                else:
                    gt_mask = self.construct_single_mask(width, height, row[2:])
                    md_mask = self.construct_single_mask(width, height, md[md_index,2:])
                    row2 = md[md_index]

                    plt.xlim([0,width])
                    plt.ylim([0,height])
                    plt.plot([row[2], row[4]], [row[3], row[3]], linewidth=1.0)
                    plt.plot([row[2], row[2]], [row[3], row[5]], linewidth=1.0)

                    plt.plot([row[4], row[2]], [row[5], row[5]], linewidth=1.0)
                    plt.plot([row[4], row[4]], [row[3], row[5]], linewidth=1.0)

                    plt.plot([row2[2], row2[4]], [row2[3], row2[3]], linewidth=1.5)
                    plt.plot([row2[2], row2[2]], [row2[3], row2[5]], linewidth=1.5)

                    plt.plot([row2[4], row2[2]], [row2[5], row2[5]], linewidth=1.5)
                    plt.plot([row2[4], row2[4]], [row2[3], row2[5]], linewidth=1.5)

                    total_gt_pixels = np.sum(gt_mask)
                    total_md_pixels = np.sum(md_mask)
                    overlap_pixels = np.sum(np.bitwise_and(gt_mask, md_mask))
                    #print "overlap: ", overlap_pixels, " gt " , total_gt_pixels, " md " , total_md_pixels
                    if overlap_pixels > 0.4 * total_gt_pixels or overlap_pixels > 0.9 * total_md_pixels:
                        plt.plot([20,50],[20,50], linewidth=2.0)
                        detected = True
                        detections[frame_gt] += 1
                        total_detected += 1
                if detected:
                    break
                else:
                    md_index += 1
                    if md_index >= md.shape[0]:
                        break
            plt.title("Frame %d"%frame_gt)
            #plt.show()
        return objects, detections

    def get_false_detections(self, gt_file, md_file, width, height):
        gt = self.read_kitti_file_with_id(gt_file)
        md = self.read_motion_detection_file_with_id(md_file)
        frame_count = gt[-1,0]

        gt_index = 0
        gt_index_start = 0
        objects = dict()
        detections = dict()

        for row in md:
            frame_md = row[0]

            if frame_md in objects:
                objects[frame_md] += 1
            else:
                objects[frame_md] = 1
            if frame_md not in detections:
                detections[frame_md] = 0


            gt_index = gt_index_start
            if gt_index >= gt.shape[0]:
                break
           
            no_detection = False

            while True:
                frame_gt = gt[gt_index,0]
                if frame_md == frame_gt:
                    gt_index_start = gt_index
                    break
                elif frame_gt > frame_md:
                    no_detection = True
                    break
                else:
                    gt_index += 1
                    if gt_index >= gt.shape[0]:
                        no_detection = True
                        gt_index_start = gt_index
                        break

            if no_detection:
                continue

            md_mask = self.construct_single_mask(width, height, row[2:])
            gt_mask = np.zeros((height, width), dtype=bool)
            while True:
                frame_gt = gt[gt_index,0]
                if frame_gt != frame_md:
                    break
                else:
                    gt_mask_obj = self.construct_single_mask(width, height, gt[gt_index,2:])
                    gt_mask = np.bitwise_or(gt_mask, gt_mask_obj)

                gt_index += 1
                if gt_index >= gt.shape[0]:
                    break

            num_overlap_pixels = np.sum(np.bitwise_and(md_mask, gt_mask))
            total_md_pixels = np.sum(md_mask)
            if num_overlap_pixels < 0.4 * total_md_pixels:
                detections[frame_md] += 1
                
        return objects, detections

    def get_stats(self, o, om, d, fd):
        total_objects = 0
        correct = 0
        false = 0
        total_detections = 0
        for key in o:
            if key not in om:
                om[key] = 0
                fd[key] = 0
            #print "Frame ", key, " objects: ", o[key], "correct detections ", d[key], " all detections: ", om[key], " false detections ", fd[key]
            total_objects += o[key]
            correct += d[key]
            false += fd[key]
            total_detections += om[key]

        return [total_objects, correct, float(correct) * 100.0 / float(total_objects), total_detections, false, float(false) * 100.0 / float(total_detections)]


rd = CompareAreas()
o,d = rd.get_detection_rate("motion_gt/0000_motion.txt", "motion_md/0000.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0000_motion.txt", "motion_md/0000.log", 1242, 375)
print "0000: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0001_motion.txt", "motion_md/0001.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0001_motion.txt", "motion_md/0001.log", 1242, 375)
print "0001: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0002_motion.txt", "motion_md/0002.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0002_motion.txt", "motion_md/0002.log", 1242, 375)
print "0002: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0003_motion.txt", "motion_md/0003.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0003_motion.txt", "motion_md/0003.log", 1242, 375)
print "0003: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0004_motion.txt", "motion_md/0004.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0004_motion.txt", "motion_md/0004.log", 1242, 375)
print "0004: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0005_motion.txt", "motion_md/0005.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0005_motion.txt", "motion_md/0005.log", 1242, 375)
print "0005: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0006_motion.txt", "motion_md/0006.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0006_motion.txt", "motion_md/0006.log", 1242, 375)
print "0006: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0007_motion.txt", "motion_md/0007.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0007_motion.txt", "motion_md/0007.log", 1242, 375)
print "0007: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0008_motion.txt", "motion_md/0008.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0008_motion.txt", "motion_md/0008.log", 1242, 375)
print "0008: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0009_motion.txt", "motion_md/0009.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0009_motion.txt", "motion_md/0009.log", 1242, 375)
print "0009: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0010_motion.txt", "motion_md/0010.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0010_motion.txt", "motion_md/0010.log", 1242, 375)
print "0010: ", rd.get_stats(o,om,d,fd)


o,d = rd.get_detection_rate("motion_gt/0011_motion.txt", "motion_md/0011.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0011_motion.txt", "motion_md/0011.log", 1242, 375)
print "0011: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0012_motion.txt", "motion_md/0012.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0012_motion.txt", "motion_md/0012.log", 1242, 375)
print "0012: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0013_motion.txt", "motion_md/0013.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0013_motion.txt", "motion_md/0013.log", 1242, 375)
print "0013: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0014_motion.txt", "motion_md/0014.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0014_motion.txt", "motion_md/0014.log", 1242, 375)
print "0014: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0015_motion.txt", "motion_md/0015.log", 1242, 375)
om,fd = rd.get_false_detections("motion_gt/0015_motion.txt", "motion_md/0015.log", 1242, 375)
print "0015: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0016_motion.txt", "motion_md/0016.log", 1224, 370)
om,fd = rd.get_false_detections("motion_gt/0016_motion.txt", "motion_md/0016.log", 1224, 370)
print "0016: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0017_motion.txt", "motion_md/0017.log", 1224, 370)
om,fd = rd.get_false_detections("motion_gt/0017_motion.txt", "motion_md/0017.log", 1224, 370)
print "0017: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0018_motion.txt", "motion_md/0018.log", 1238, 374)
om,fd = rd.get_false_detections("motion_gt/0018_motion.txt", "motion_md/0018.log", 1238, 374)
print "0018: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0019_motion.txt", "motion_md/0019.log", 1238, 374)
om,fd = rd.get_false_detections("motion_gt/0019_motion.txt", "motion_md/0019.log", 1238, 374)
print "0019: ", rd.get_stats(o,om,d,fd)

o,d = rd.get_detection_rate("motion_gt/0020_motion.txt", "motion_md/0020.log", 1241, 376)
om,fd = rd.get_false_detections("motion_gt/0020_motion.txt", "motion_md/0020.log", 1241, 376)
print "0020: ", rd.get_stats(o,om,d,fd)

'''
total_objects = 0
correct = 0
false = 0
total_detections = 0
for key in o:
    if key not in om:
        om[key] = 0
        fd[key] = 0
    #print "Frame ", key, " objects: ", o[key], "correct detections ", d[key], " all detections: ", om[key], " false detections ", fd[key]
    total_objects += o[key]
    correct += d[key]
    false += fd[key]
    total_detections += om[key]

print "total: ", total_objects
print "correct: ", correct
print "percentage: " , float(correct) / float(total_objects)
print "false: " , false
print "total_det: ", total_detections
print "percentage: ", float(false) / float(total_detections)
'''

'''
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
'''
