"""
Author: Omkar Damle
March 2018
"""

from torch.utils.data import Dataset
import os
import numpy as np
import cv2
from scipy.misc import imresize

class DAVIS17OnlineDataset(Dataset):

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None, noIterations=1,
                 object_id = -1):
        """Loads deformations along with images and ground truth examples
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations" and "Deformations"
        """
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name
        self.object_id = object_id

        image_list_fname = 'train'

        angle_list = [2*(angle+1) for angle in range(10)]
        neg_angle_list = [-angle for angle in angle_list]
        angle_list.extend(neg_angle_list)
        angle_list.extend([None])

        flip_list = [0,1]

        if self.seq_name is None:
            print('Please give sequence name')
            return
        else:
            image_list = []
            gt_list = []
            deformations1_list = []
            deformations2_list = []

            base_path = "Deformations/480p/" + self.seq_name + "_online/00000"

            for iterNo in range(noIterations):
                for angle in angle_list:
                    for flip in flip_list:
                        angleString = ""
                        if angle is not None:
                            angleString = 'angle' + str(angle) + '_'

                        flipString = ""
                        if flip==1:
                            flipString = 'flipped_'

                        tempString = base_path + '_' + str(iterNo+1) + '_' + angleString + flipString
                        tempString1 = base_path + '_' + str(self.object_id) + '_' + str(iterNo+1) + '_' + angleString + flipString

                        image_list.append(tempString + 'i.png')
                        gt_list.append(tempString1 + 'gt.png')
                        deformations1_list.append(tempString1 + 'd1.png')
                        deformations2_list.append(tempString1 + 'd2.png')

        self.img_list = image_list
        self.labels = gt_list
        self.deformations1 = deformations1_list
        self.deformations2 = deformations2_list

        assert (len(self.img_list) == len(self.labels))
        assert(len(self.labels) == len(self.deformations1))
        assert(len(self.labels) == len(self.deformations2))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img, gt = self.make_img_gt_pair(idx)

        df1, df2 = self.make_df1_df2(idx)

        sample = {'image': img, 'gt': gt, 'df1': df1, 'df2':df2}

        if self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
            if self.labels[idx] is not None:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
                gt = np.array(label, dtype=np.float32)
                gt = gt/np.max([gt.max(), 1e-8])

        return img, gt

    def make_df1_df2(self, idx):
        """
        Make the deformations
        """
        df1 = cv2.imread(os.path.join(self.db_root_dir, self.deformations1[idx]), cv2.IMREAD_GRAYSCALE)
        df2 = cv2.imread(os.path.join(self.db_root_dir, self.deformations2[idx]), cv2.IMREAD_GRAYSCALE)

        if self.inputRes is not None:
            df1 = imresize(df1, self.inputRes, interp='nearest')
            df2 = imresize(df2, self.inputRes, interp='nearest')

        df1 = np.array(df1, dtype=np.float32)
        df1 = df1/np.max([df1.max(), 1e-8])

        df2 = np.array(df2, dtype=np.float32)
        df2 = df2/np.max([df2.max(), 1e-8])

        return df1, df2
