"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from PIL import Image
import torch
import numpy as np
import glob
import torchvision
import random
from torch.utils.data import DataLoader, Dataset
import cv2

class KittiDepthDataset(Dataset):
    def __init__(self, data_path, gt_path, setname='train', transform=None, norm_factor=256, invert_depth=False,
                 rgb_dir=None, rgb2gray=False, fill_depth = False, flip = False, blind = False):
        self.data_path = data_path
        self.gt_path = gt_path
        self.setname = setname
        self.transform = transform
        self.norm_factor = norm_factor
        self.invert_depth = invert_depth
        self.rgb_dir = rgb_dir
        self.rgb2gray = rgb2gray
        self.fill_depth = fill_depth
        self.flip = flip
        self.blind = blind
        self.i = 0
        print("Depth dir: ", self.data_path)
        print("RGB Dir: ", self.rgb_dir)
        self.data = list(sorted(glob.iglob(self.data_path + "/*.png", recursive=True)))
        print("Number of depth images: ", len(self.data))
        # self.gt = list(sorted(glob.iglob(self.gt_path + "/**/*.png", recursive=True)))
        self.rgb = list(sorted(glob.iglob(self.rgb_dir + "/*.png", recursive=True)))
        # assert (len(self.gt) == len(self.data))
        print("Number of rgb images: ", len(self.rgb))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        # Check if Data filename is equal to GT filename
        # if self.setname == 'train' or self.setname == 'val':
        #     data_path = self.data[item].split('/')[7]
        #     # print(data_path)
        #     # gt_path = self.gt[item].split(self.setname)[1]
        #     # print(type(self.data[item]))
        #     # assert (data_path[0:25] == gt_path[0:25])  # Check folder name

        #     # data_path = data_path.split('/')[2]
        #     # # gt_path = gt_path.split('/')[2]
        #     # print(data_path)
        #     # data_path = data_path.split('.')[0]
        #     # gt_path = gt_path.split('.')[0]
        #     print("NEW DATA PATH: ", data_path)
            # assert (data_path == gt_path)  # Check filename

            # Set the certainty path
            # sep = str(self.data[item])
            # sep = str(self.data[item]).split('/sun_depth/')
            # print("SEP VALUE ", sep)

        # elif self.setname == 'selval':
        #     data_path = self.data[item].split('00000')[1]
        #     gt_path = self.gt[item].split('00000')[1]
        #     assert (data_path == gt_path)
        #     # Set the certainty path
        #     sep = str(self.data[item]).split('/velodyne_raw/')


        # print("length depth: ", len(self.data))
        # Read images and convert them to 4D floats
        data = Image.open(str(self.data[item]))
        # gt = Image.open(str(self.gt[item]))



        # Read RGB images
        if self.setname == 'train' or self.setname == 'val':

            # gt_path = str(self.gt[item])
            # idx = gt_path.find('/ngr_5classes')
            # print(idx)
            # fname = gt_path[idx:idx + 10]
            # print(fname)
            # print("len rgb: ", len(self.rgb))
            # print(type(self.rgb))
            rgb_path = self.rgb[item]
            # self.rgb[:, :, [0, 2]] = self.rgb[:, :, [2, 0]]
            rgb = Image.open(str(rgb_path))
            # width, height = rgb.size
            # for x in range(0, width):
            #      for y in range(0,height):
            #         r, g, b = rgb.getpixel((x, y))
            #         rgb.putpixel((x, y), (b, g, r))
                    # cv2.imwrite("/home/core_uc/rgn/image.png", (b,g,r))
                    # cv2.imshow('test', np.array([b,g,r]))
                    # cv2.waitKey(2)
        # elif self.setname == 'selval':
        #     data_path = str(self.data[item])
        #     idx = data_path.find('velodyne_raw')
        #     fname = data_path[idx + 12:]
        #     idx2 = fname.find('velodyne_raw')
        #     rgb_path = data_path[:idx] + 'image' + fname[:idx2] + 'image' + fname[idx2 + 12:]
        #     rgb = Image.open(rgb_path)
        # elif self.setname == 'test':
        #     data_path = str(self.data[item])
        #     idx = data_path.find('velodyne_raw')
        #     fname = data_path[idx + 12:]
        #     rgb_path = data_path[:idx] + 'image/' + fname
        #     rgb = Image.open(rgb_path)





        if self.rgb2gray:
            t = torchvision.transforms.Grayscale(1)
            rgb = t(rgb)

        W, H = data.size

        # print(W, H)



        # Apply transformations if given
        if self.transform is not None:
            data = self.transform(data)
            # gt = self.transform(gt)
            rgb = self.transform(rgb)
        # if self.transform is None and self.setname == 'train':
        #     crop_lt_u = random.randint(0, W - 1216)
        #     crop_lt_v = random.randint(0, H - 352)
        #     data = data.crop((crop_lt_u, crop_lt_v, crop_lt_u+1216, crop_lt_v+352))
        #     gt = gt.crop((crop_lt_u, crop_lt_v, crop_lt_u + 1216, crop_lt_v + 352))
        #     rgb = rgb.crop((crop_lt_u, crop_lt_v, crop_lt_u + 1216, crop_lt_v + 352))


        # if self.flip and random.randint(0, 1) and self.setname == 'train':
        #     data = data.transpose(Image.FLIP_LEFT_RIGHT)
        #     gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        #     rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to numpy
        data = np.array(data, dtype=np.float64)
        # gt = np.array(gt, dtype=np.float16)
        # w, h = data.shape
        # y = int(h/3) 
        # # Create new image with only the bottom half of the segmentation mask
        # mask = cv2.imread("/home/core_uc/depth_completion/datasets/depth_pilot/depth/depth_1.png", cv2.IMREAD_ANYDEPTH)
        # print(mask.shape)
        # mask = cv2.resize(mask, (W,H))
        # # mask = cv2.rectangle(mask, (0,y), (h,w), (255,255,255), cv2.FILLED)
        # data = cv2.bitwise_and(data, mask)
        # cv2.imread('test', data)
        # cv2.waitKey(2)
        # for x in range(0, w ):
        #      for y in range(0,h):
        #         if x % 2 == 0 or y % 2 == 0:
        #             data[x,y] = 0

        #blind
        # if self.blind and (self.setname == 'train'):
        #     blind_start = random.randint(100, H - 50)
        #     data[blind_start:blind_start+50, :] = 0

        # define the certainty
        C = (data > 0).astype(float)

        print('*'*80)
        # Normalize the data
        # data =  cv2.normalize(src = data, dst = None, alpha = 0, beta = 65535, 
                    # norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_16U)
        # data *= 1000
        # cv2.imwrite("/home/core_uc/test_" + str(self.i) + ".png", data)
        self.i+=1
        print("Depth max/min value before normalization: ", data.max(), data.min())
        print("Depth Shape: ", data.shape)
        data = data / self.norm_factor  # [0,1]
        # data =  cv2.normalize(src = data, dst = None, alpha = 0, beta = 70, 
                    # norm_type=cv2.NORM_MINMAX)
        print("Depth max/min value after normalization: ", data.max(), data.min())
        # gt = gt / self.norm_factor

        # Expand dims into Pytorch format
        data = np.expand_dims(data, 0)
        # print("Depth shape: ", data.shape)
        # gt = np.expand_dims(gt, 0)
        C = np.expand_dims(C, 0)

        # Convert to Pytorch Tensors
        data = torch.tensor(data, dtype=torch.float)
        # gt = torch.tensor(gt, dtype=torch.float)
        C = torch.tensor(C, dtype=torch.float)

        # Convert depth to disparity
        if self.invert_depth:
            data[data == 0] = -1
            data = 1 / data
            data[data == -1] = 0

            # gt[gt == 0] = -1
            # gt = 1 / gt
            # gt[gt == -1] = 0

        # Convert RGB image to tensor

        rgb = np.array(rgb, dtype=np.float16)
        # print("RGB max value before normalization: ", rgb.max())
        # print("RGB Shape: ", rgb.shape)

        rgb /= 255
        # print("RGB max value after normalization: ", rgb.max())

        if self.rgb2gray:
            rgb = np.expand_dims(rgb, 0)
        else:
            rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.tensor(rgb, dtype=torch.float)
        gt = 0
        return data, C, gt, item, rgb
