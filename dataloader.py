from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import sys
import cv2
import numpy as np
import glob
import os,json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tr
import math


def resize_image(inp_img, target_size):

    out_img = cv2.resize(inp_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return out_img


def random_crop_flip(inp_img, out_img):
    """
    Function to randomly crop and flip images.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image.
    :return: The randomly cropped and flipped image.
    """
    h, w = out_img.shape

    rand_h = np.random.randint(h/8)
    rand_w = np.random.randint(w/8)
    offset_h = 0 if rand_h == 0 else np.random.randint(rand_h)
    offset_w = 0 if rand_w == 0 else np.random.randint(rand_w)
    p0, p1, p2, p3 = offset_h, h+offset_h-rand_h, offset_w, w+offset_w-rand_w

    rand_flip = np.random.randint(10)
    if rand_flip >= 5:
        inp_img = inp_img[::, ::-1, ::]
        out_img = out_img[::, ::-1]

    return inp_img[p0:p1, p2:p3], out_img[p0:p1, p2:p3]


def random_rotate(inp_img, out_img, max_angle=25):
    """
    Function to randomly rotate images within +max_angle to -max_angle degrees.
    This algorithm does NOT crops the edges upon rotation.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image.
    :param max_angle: Maximum angle an image can be rotated in either direction.
    :return: The randomly rotated image.
    """
    angle = np.random.randint(-max_angle, max_angle)
    h, w = out_img.shape
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute new dimensions of the image and adjust the rotation matrix
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(inp_img, M, (new_w, new_h)), cv2.warpAffine(out_img, M, (new_w, new_h))


def random_rotate_lossy(inp_img, out_img, max_angle=25):
    """
    Function to randomly rotate images within +max_angle to -max_angle degrees.
    This algorithm crops the edges upon rotation.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image.
    :param max_angle: Maximum angle an image can be rotated in either direction.
    :return: The randomly rotated image.
    """
    angle = np.random.randint(-max_angle, max_angle)
    h, w = out_img.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(inp_img, M, (w, h)), cv2.warpAffine(out_img, M, (w, h))


def random_brightness(inp_img):
    """
    Function to randomly perturb the brightness of the input images.
    :param inp_img: A H x W x C input image.
    :return: The image with randomly perturbed brightness.
    """
    contrast = np.random.rand(1) + 0.5
    light = np.random.randint(-20, 20)
    inp_img = contrast * inp_img + light

    return np.clip(inp_img, 0, 255)



class dataLoader(Dataset):
    """
    DataLoader for training.
    """
    def __init__(self, img_path, label_path, augment_data=False, target_size=256):
        print(img_path) 
        print(label_path)
        if os.path.exists(img_path) and os.path.exists(label_path):
            self.inp_path = img_path
            self.out_path = label_path
        else:
            print("Please check the input and output path!")
            sys.exit(0)
        self.augment_data = augment_data
        self.target_size = target_size
        self.inp_files = os.listdir(self.inp_path)


    def __getitem__(self, idx):
        
        img_path = os.path.join(self.inp_path, self.inp_files[idx])
        label_path = os.path.join(self.out_path, self.inp_files[idx])

        if os.path.exists(img_path) and os.path.exists(label_path):

            inp_img = cv2.imread(img_path)
            inp_img = cv2.resize(inp_img, ( self.target_size,  self.target_size), interpolation=cv2.INTER_LINEAR)

            #inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            mask_img = cv2.imread(label_path, 0)
            mask_img = cv2.resize(mask_img, ( self.target_size,  self.target_size), interpolation=cv2.INTER_LINEAR)

            _, mask_img = cv2.threshold(mask_img,127.5,255,cv2.THRESH_BINARY)
            
            #inp_img, mask_img = resize_image(inp_img, self.target_size), resize_image(mask_img, self.target_size)

            mask_img = mask_img.astype('float32')
            inp_img = inp_img.astype('float32')

            # if self.augment_data:
            #     inp_img, mask_img = random_crop_flip(inp_img, mask_img)
            #     inp_img, mask_img = random_rotate(inp_img, mask_img)
            #     inp_img = random_brightness(inp_img)

            inp_img /= 255.0
            inp_img = np.transpose(inp_img, axes=(2, 0, 1))
            mask_img /= 255.0
            #mask_img = toLabel(mask_img)
        else:
            print("Please check the images and labels!")
            print(img_path)
            sys.exit(0)

        return torch.from_numpy(inp_img).float(), torch.from_numpy(mask_img).long()

    def __len__(self):
        return len(self.inp_files)


class dataLoader_sp(Dataset):
    """
    DataLoader for training.
    """
    def __init__(self, img_path, label_path, augment_data=False, target_size=256):
        print(img_path) 
        print(label_path)
        if os.path.exists(img_path) and os.path.exists(label_path):
            self.inp_path = img_path
            self.out_path = label_path
        else:
            print("Please check the input and output path!")
            sys.exit(0)
        self.augment_data = augment_data
        self.target_size = target_size
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                       std=[0.229, 0.224, 0.225])

        # self.transform = transforms.Compose([
        #     transforms.Resize([256, 256]),
        #     transforms.ToTensor()]) 

        self.inp_files = os.listdir(img_path)


    def __getitem__(self, idx):
        
        img_path = os.path.join(self.inp_path, self.inp_files[idx])
        label_path = os.path.join(self.out_path, self.inp_files[idx])

        if os.path.exists(img_path) and os.path.exists(label_path):

            inp_img = cv2.imread(img_path)
            inp_img = cv2.resize(inp_img, ( self.target_size,  self.target_size), interpolation=cv2.INTER_LINEAR)

            #inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            mask_img = cv2.imread(label_path, 0)
            mask_img = cv2.resize(mask_img, ( self.target_size,  self.target_size), interpolation=cv2.INTER_LINEAR)

            _, mask_img = cv2.threshold(mask_img,127.5,255,cv2.THRESH_BINARY)
            
            #inp_img, mask_img = resize_image(inp_img, self.target_size), resize_image(mask_img, self.target_size)

            mask_img = mask_img.astype('float32')
            inp_img = inp_img.astype('float32')

            # if self.augment_data:
            #     inp_img, mask_img = random_crop_flip(inp_img, mask_img)
            #     inp_img, mask_img = random_rotate(inp_img, mask_img)
            #     inp_img = random_brightness(inp_img)

            inp_img /= 255.0
            inp_img = np.transpose(inp_img, axes=(2, 0, 1))
            mask_img /= 255.0
            #mask_img = toLabel(mask_img)
        else:
            print("Please check the images and labels!")
            print(img_path)
            sys.exit(0)

        return torch.from_numpy(inp_img).float(), torch.from_numpy(mask_img).long(),self.inp_files[idx]

    def __len__(self):
        return len(self.inp_files)

class InfDataloader(Dataset):
    """
    Dataloader for Inference.
    """
    def __init__(self, img_folder, target_size=256):
        self.imgs_folder = img_folder
        self.img_paths = sorted(glob.glob(self.imgs_folder + '/*'))

        self.target_size = target_size
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                       std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        """
        __getitem__ for inference
        :param idx: Index of the image
        :return: img_np is a numpy RGB-image of shape H x W x C with pixel values in range 0-255.
        And img_tor is a torch tensor, RGB, C x H x W in shape and normalized.
        """
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pad images to target size
        img_np = resize_image(img, self.target_size)
        img_tor = img_np.astype(np.float32)
        img_tor = img_tor / 255.0
        img_tor = np.transpose(img_tor, axes=(2, 0, 1))
        img_tor = torch.from_numpy(img_tor).float()
        #img_tor = self.normalize(img_tor)

        return img_np, img_tor

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='config path')
    args = parser.parse_args()
    file_dir = args.config_path

    with open(file_dir) as f:
            config = json.load(f)
    print(config)
    img_size = config['DATA']['image_size']
    bs = config['TRAIN']['batch_size']
    train_img = os.path.join(config['DATA']['data_dir'],config['DATA']['train_dir'])
    train_label = os.path.join(config['DATA']['data_dir'],config['DATA']['cam_dir'])
    test_img = os.path.join(config['DATA']['data_dir'],config['DATA']['test_dir'])
    test_label = os.path.join(config['DATA']['data_dir'],config['DATA']['test_gt'])
    train_data = dataLoader(img_path=train_img, label_path=train_label, augment_data=False, target_size=img_size)
    val_data = dataLoader(img_path=test_img, label_path=test_label, augment_data=False, target_size=img_size)
    test_data = InfDataloader(test_img, target_size=img_size)

    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=2)

    print("Train Dataloader :")
    for batch_idx, (inp_imgs, gt_masks) in enumerate(train_dataloader):
        print('Loop :', batch_idx, inp_imgs.size(), gt_masks.size())
        if batch_idx == 3:
            break

    print("\nTest Dataloader :")
    for batch_idx, (inp_imgs, gt_masks) in enumerate(test_dataloader):
        print('Loop :', batch_idx, inp_imgs.size(), gt_masks.size())
        if batch_idx == 3:
            break

    # # Test image augmentation functions
    # inp_img = cv2.imread('./data/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00000003.jpg')
    # out_img = cv2.imread('./data/DUTS/DUTS-TE/DUTS-TE-Mask/ILSVRC2012_test_00000003.png', -1)
    # # inp_img = inp_img.astype('float32')
    # out_img = out_img.astype('float32')
    # out_img = out_img / 255.0

    # cv2.imshow('Original Input Image', inp_img)
    # cv2.imshow('Original Output Image', out_img)

    # print('\nImage shapes before processing :', inp_img.shape, out_img.shape)
    # x, y = random_crop_flip(inp_img, out_img)
    # x, y = random_rotate(x, y)
    # x = random_brightness(x)
    # x, y = resize_image(x, target_size=256), resize_image(y, target_size=256)
    # # x now contains float values, so either round-off the values or convert the pixel range to 0-1.
    # x = x / 255.0
    # print('Image shapes after processing :', x.shape, y.shape)

    # cv2.imshow('Processed Input Image', x)
    # cv2.imshow('Processed Output Image', y)
    # cv2.waitKey(0)