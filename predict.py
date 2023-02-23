import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
#from torchsummary import summary
from torch.autograd import Variable
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from dataloader import dataLoader, InfDataloader
from utils import mkdir,model_train,setup_seed
from model import RAN,Att_RAN,Att_Acc_RAN
import os, json
# from network import WSDDN
from tensorboardX import SummaryWriter
#from loss import loss_ce
from easydict import EasyDict as edict
import time
from torchsummary import summary
from torch.optim import lr_scheduler
from loss import WCE
import cv2
import numpy as np
from scipy.io import savemat


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    setup_seed(20)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='config path')
    args = parser.parse_args()
    file_dir = args.config_path

    with open(file_dir) as f:
            config = json.load(f)

    config = edict(config)

    dataset = config['DATA']['dataset']
    img_size = config['DATA']['image_size']
    bs = config['TRAIN']['batch_size']
    lr = config['TRAIN']['lr']
    weight_decay = config['TRAIN']['weight_decay']
    lr_decay = config['TRAIN']['lr_decay']
    image_size = config['DATA']['image_size']


    train_img = os.path.join(config['DATA']['data_dir'],config['DATA']['train_dir'])
    train_label = os.path.join(config['DATA']['data_dir'],config['DATA']['cam_dir'])
    test_img = os.path.join(config['DATA']['data_dir'],config['DATA']['test_dir'])
    test_label = os.path.join(config['DATA']['data_dir'],config['DATA']['test_gt'])

    train_data = dataLoader(img_path=train_img, label_path=train_label, augment_data=False, target_size=img_size)
    val_data = dataLoader(img_path=test_img, label_path=test_label, augment_data=False, target_size=img_size)
    test_data = InfDataloader(test_img, target_size=img_size)

    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=0)

    model_path = config.TEST.model_path

    model = RAN().to(device)
    model.load_state_dict(torch.load(model_path))

    image_list = os.listdir(test_img)
    save_dir = config.TEST.save_dir

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        #print('!!!!!!!!!!!!!!!!!!!')

    for image_name in image_list:
        inp_img = cv2.imread(os.path.join(test_img, image_name)).astype('float32')
        inp_img = cv2.resize(inp_img, ( image_size,  image_size), interpolation=cv2.INTER_LINEAR)

        inp_img /= 255.0
        inp_img = np.transpose(inp_img, axes=(2, 0, 1))
        inp_img = torch.from_numpy(inp_img).float().to(device)
        inp_img = inp_img.unsqueeze(0)
        pred_masks = model(inp_img)
        pre = pred_masks[0,1,:,:].cpu().detach().numpy()
        cv2.imwrite(os.path.join(save_dir, image_name), pre*255)

