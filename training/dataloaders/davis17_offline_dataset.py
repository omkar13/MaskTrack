"""
Author: Omkar Damle
Date: April 2018
"""
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from PIL import Image
import cv2

class DAVIS17Offline(Dataset):
    def __init__(self, train=True, mini=False, mega=False,
                 inputRes=None,
                 db_root_dir='DAVIS17',
                 transform=None):

        self.train = train
        self.mini = mini
        self.mega = mega
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform

        if self.mini == False and self.mega == False:
            if self.train:
                fname = 'train'
                # fname = 'train_seqs'
            else:
                fname = 'val'
                # fname = 'val_seqs'
        elif self.mini == True:
            if self.train:
                fname = 'train_mini'
                # fname = 'train_seqs'
            else:
                fname = 'val_mini'
                # fname = 'val_seqs'
        elif self.mega == True:
            if self.train:
                fname = 'train_mega'
            else:
                fname = 'val_mega'

        img_list = []
        labels = []
        deformations = []

        # Initialize the original DAVIS splits for training the parent network
        with open(os.path.join(db_root_dir, 'ImageSets/2017/' + fname + '.txt')) as f:
            seqs = f.readlines()

            for seq in seqs:
                images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))

                no_objects = len(glob.glob(os.path.join(db_root_dir, 'Annotations_binary/480p',seq.strip(),'00000_*.png')))

                for image in images:
                    image_id = image.split('.')[0]
                    for object_id in range(1,no_objects+1):
                        for df in [1,2]:
                            img_list.append(os.path.join('JPEGImages/480p',seq.strip(),image))
                            labels.append(os.path.join('Annotations_binary/480p', seq.strip(), image_id + '_' + str(object_id) + '.png'))
                            deformations.append(os.path.join('Deformations/480p', seq.strip(), image_id + '_' + str(object_id) + '_d' + str(df) + '.png'))

        assert (len(labels) == len(img_list))
        assert (len(labels) == len(deformations))

        self.img_list = img_list
        self.deformations = deformations
        self.labels = labels

        print('Done initializing ' + fname + ' Dataset')

    def __getitem__(self, idx):

        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        label = Image.open(os.path.join(self.db_root_dir, self.labels[idx]))
        deformation = Image.open(os.path.join(self.db_root_dir, self.deformations[idx]))
        #print(os.path.join(self.db_root_dir, self.deformations[idx]))
        img,label,deformation = self.transform(np.array(img), np.array(label), np.array(deformation), self.inputRes)
        sample = {'image': img, 'gt': label, 'deformation': deformation}
        return sample

    def __len__(self):
        return len(self.img_list)