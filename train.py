import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
#from torchsummary import summary
from torch.autograd import Variable
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from dataloader import dataLoader, InfDataloader,dataLoader_sp
from utils import mkdir,setup_seed,model_train,model_train_sp
#from loss import ModelLossSemsegGatedCRF
from model import RAN
#from deeplabv3 import DeepLabV3
import os, json
# from network import WSDDN
from tensorboardX import SummaryWriter
#from loss import loss_ce
from easydict import EasyDict as edict
import time
from torchsummary import summary
from torch.optim import lr_scheduler
from loss_t import WCE,ModelLossSemsegGatedCRF,WCE_sp
import shutil
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device=torch.device('cpu')

if __name__ == '__main__':

    setup_seed(20)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='config path')
    args = parser.parse_args()
    file_dir = args.config_path

    with open(file_dir) as f:
            config = json.load(f)

    config = edict(config)

    rgb = config['TRAIN']['rgb']
    xy = config['TRAIN']['xy']
    xy2 = config['TRAIN']['xy2']
    dataset = config['DATA']['dataset']
    img_size = config['DATA']['image_size']
    bs = config['TRAIN']['batch_size']
    lr = config['TRAIN']['lr']
    weight_decay = config['TRAIN']['weight_decay']
    lr_decay = config['TRAIN']['lr_decay']
    wce = config['TRAIN']['wce']
    weight = config['TRAIN']['weight']

    CRFconfig = [{'weight': weight, 'xy': xy,'rgb': rgb},{'weight': 1-weight,'xy': xy2}]

    train_img = os.path.join(config['DATA']['data_dir'],config['DATA']['train_dir'])
    train_label = os.path.join(config['DATA']['data_dir'],config['DATA']['cam_dir'])
    test_img = os.path.join(config['DATA']['data_dir'],config['DATA']['test_dir'])
    test_label = os.path.join(config['DATA']['data_dir'],config['DATA']['test_gt'])


    train_data = dataLoader(img_path=train_img, label_path=train_label, augment_data=False, target_size=256)
    val_data = dataLoader(img_path=test_img, label_path=test_label, augment_data=False, target_size=256)
    test_data = InfDataloader(test_img, target_size=256)

    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=0)


    time_now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    model_path = os.path.join('./checkpoint/',dataset, time_now)
    log_dir = os.path.join('./log/',dataset, time_now)
    save_dir = os.path.join('./result/',dataset, time_now,'train')

    if not os.path.exists(model_path):
        mkdir(model_path)
    if not os.path.exists(log_dir):
        mkdir(log_dir)
    if not os.path.exists(save_dir):
        mkdir(save_dir)

    config.TEST.model_path = model_path
    config.TEST.save_dir = save_dir

    with open(file_dir, 'w') as f:
             f.write(json.dumps(config, indent=4))
    
    with open(os.path.join(log_dir,'config.json'),'w') as f:
        f.write(json.dumps(config, indent=4))

    #model = Att_Acc_RAN().to(device)
    model = RAN().to(device)
    #summary(model, (3, 256, 256))
    model.load_state_dict(torch.load('./checkpoint/ACT/20230222162158/best-model_epoch-124.pth'))
    #criterion1 =  ModelLossSemsegGatedCRF().to(device)
    #print("model model model ")
    criterion2 =  ModelLossSemsegGatedCRF().to(device)
    criterion3 = WCE().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

    writer = SummaryWriter(log_dir)
 
    if config['TRAIN']['self_pace']:
        print('********************Self-pace Learning*****************')
        criterion1 = WCE_sp().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=weight_decay)
        model_train_sp(model, val_dataloader,criterion1,criterion2,criterion3,optimizer, writer, model_path, device, config,CRFconfig)

    else:
        
        print('********************First Stage*****************')
        criterion1 = WCE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model_train(model, train_dataloader, val_dataloader,criterion1,criterion2,criterion3,optimizer, writer, model_path, device, config,CRFconfig)
    #model_train_un(model, train_dataloader, val_dataloader,criterion1,criterion2,criterion3,optimizer, scheduler, writer, model_path, device, config,CRFconfig)

