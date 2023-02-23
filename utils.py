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
from utils import *
import os, json
import numpy as np
import math
from tensorboardX import SummaryWriter
#from loss import loss_ce
from easydict import EasyDict as edict
from datetime import datetime
from torchsummary import summary
import cv2
from torch.optim import lr_scheduler
import random

from torch.autograd import Variable
import itertools


def toLabel(input):
    temp = 1-input 
    newlabel = torch.cat((input, temp), 0)
    return newlabel


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate



def model_val(model,val_dataloader,criterion, ep, writer, device, config):
    
    model.eval()
    img_size = config['DATA']['image_size']
    tot_loss = 0
    tp_tf = 0   # TruePositive + TrueNegative, for accuracy
    tp = 0      # TruePositive
    pred_true = 0   # Number of '1' predictions, for precision
    gt_true = 0     # Number of '1's in gt mask, for recall
    mae_list = []   # List to save mean absolute error of each image

    with torch.no_grad():
        for batch_idx, (inp_imgs, gt_masks) in enumerate(val_dataloader, start=1):
            if batch_idx>10: 
                break

            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)

            pred_masks = model(inp_imgs)
            loss = criterion(pred_masks, gt_masks.long())

            tot_loss += loss.item()

            mask_img = pred_masks.cpu().numpy()
            mask = mask_img[:,1,:,:]
            mask[mask>=0.5] = 1
            mask[mask<0.5] = 0
            gts = gt_masks[:,:,:].cpu().numpy()
            
            for  i in range(len(inp_imgs)):
                gt =  gts[i,:,:]
                mask.astype(np.float)
                gt.astype(np.float)
                tp_tf += np.sum(mask == gt)
                tp += np.multiply(mask, gt).sum()
                pred_true += np.sum(mask)
                gt_true += np.sum(gt)
            

    avg_loss = tot_loss / batch_idx
    accuracy = tp_tf / (len(val_dataloader) *img_size * img_size)
    precision = tp / (pred_true + 1e-5)
    recall = tp / (gt_true + 1e-5)
    F = 2./(1./precision + 1./recall)

    writer.add_scalar('Val/loss', avg_loss, ep)
    writer.add_scalar('Val/accuracy', accuracy, ep)
    writer.add_scalar('Val/precision', precision, ep)
    writer.add_scalar('Val/recall', recall, ep)
    writer.add_scalar('Val/F', F, ep)

    
    writer.add_image('val/input', inp_imgs[1,:,:,:].squeeze(0), ep)
    writer.add_image('val/output', pred_masks[1,1,:,:].unsqueeze(0), ep)
    writer.add_image('val/gt', gt_masks[1,:,:].unsqueeze(0), ep)

    # the results may not be accurate, just for observe the model training process
    print('val :: ACC : {:.4f}\tPRE : {:.4f}\tREC : {:.4f}\tF : {:.4f}\tAVG-LOSS : {:.4f}\n'.format(
                                                                                            accuracy,
                                                                                            precision,
                                                                                            recall,
                                                                                            F,
                                                                                            avg_loss))

    return avg_loss


def model_train(model, train_dataloader, val_dataloader, criterion1, criterion2,criterion3,optimizer, writer, model_path, device, config,CRFconfig):
    k = 0
    best_test_mae = float('inf')
    epoch = config['TRAIN']['first_epoch_num']
    log_interval = config['TRAIN']['log_interval']
    val_interval = config['VAL']['val_interval']
    crf_w = config['TRAIN']['crf_w']
    wce = config['TRAIN']['wce']
    decay_interval = config['TRAIN']['decay_interval']

    for ep in range(epoch):

        model.train()

        #torch.autograd.set_detect_anomaly(True)

        #print(epoch)
        for batch_idx, (inp_imgs, gt_masks) in enumerate(train_dataloader):

            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            #print('input_img:',inp_imgs.shape)
            optimizer.zero_grad()
            pred_masks= model(inp_imgs)
            
            loss1 = criterion1(pred_masks, gt_masks)
            loss2 = criterion2(pred_masks[:,1,:,:].unsqueeze(1), CRFconfig,inp_imgs)
            loss = loss1+crf_w*loss2
            

            with torch.autograd.detect_anomaly():
                loss.backward()

            optimizer.step()
            # for p in model.parameters():
            #     if p.requires_grad:
            #         print(p.name, p.data.max())
            # for p in optimizer.param_groups[0]['params']:
            #     print(p.grad)
                #print(x.name, x.grad)
            if batch_idx % log_interval == 0:
                k +=1
                print('{} TRAIN :: Epoch : {}\tBatch : {}/{} ({:.2f}%)\t\tTot Loss : {:.4f}'
                        .format(datetime.now(),ep + 1, batch_idx + 1, len(train_dataloader), (batch_idx + 1) * 100 / len(train_dataloader),loss.item()))
                writer.add_scalar('Train/loss', loss.item(), k)
                #writer.add_scalar('Train/celoss', loss1.item(), k)
                writer.add_scalar('Train/crfloss', loss2, k)
                #writer.add_scalar('Train/tfsloss', loss3.item(), k)
                writer.add_image('Train/input', torch.squeeze(inp_imgs[1,:,:,:]), k)
                #writer.add_image('Train/logvar', logvar[1,0,:,:].unsqueeze(0), k)
                #_image('Train/trans_inp', torch.squeeze(trans_inp[1,:,:,:]), k)
                writer.add_image('Train/output', pred_masks[1,1,:,:].unsqueeze(0), k)
                # writer.add_image('Train/trans_out', trans_mask[1,0,:,:].unsqueeze(0), k)
                # writer.add_image('Train/trans_pred', trans_pred[1,1,:,:].unsqueeze(0), k)

                writer.add_image('Train/gt', gt_masks[1,:,:].unsqueeze(0), k)
                writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], k)


            if batch_idx % decay_interval == 0 and optimizer.param_groups[0]['lr']>1e-5:
                adjust_learning_rate(optimizer, decay_rate=.9)
        # Validation
        if ep % val_interval == 0:
            model.eval()
            val_mae = model_val(model,val_dataloader,criterion3, ep, writer, device, config)

        if ep % 4 == 0 and ep>0:
            torch.save(model.state_dict(), os.path.join(model_path, 'best-model_epoch-{:03}.pth'.
                            format(ep)))

def generate_new_label(prediction,labelname, label_update_dir):
    predict = prediction*255
    label= predict.detach().cpu().numpy().astype('uint8')
    _, gt = cv2.threshold(label,127.5,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    newlabel = cv2.morphologyEx(gt, cv2.MORPH_OPEN, kernel)
    #print(newlabel.shape)
    cv2.imwrite(os.path.join("E:\Jue\RSE2023\Methods\SP-RAN/torch\labels",labelname),newlabel)


def model_train_sp(model, val_dataloader, criterion1, criterion2,criterion3,optimizer, writer, model_path, device, config,CRFconfig):
    k = 0
    best_test_mae = float('inf')
    epoch = config['TRAIN']['self_paced_epoch_num']
    log_interval = config['TRAIN']['log_interval']
    val_interval = config['VAL']['val_interval']
    crf_w = config['TRAIN']['crf_w']
    wce = config['TRAIN']['wce']
    decay_interval = config['TRAIN']['decay_interval']
    label_update_dir = config['TRAIN']['label_update_dir']
    new_train_list = os.listdir(os.path.join(config['DATA']['data_dir'],config['DATA']['train_dir']))
    lamda = config['TRAIN']['lamda']
    train_img = os.path.join(config['DATA']['data_dir'],config['DATA']['train_dir'])
    train_label = os.path.join(config['DATA']['data_dir'],config['DATA']['cam_dir'])
    bs = config['TRAIN']['batch_size']



    train_data = dataLoader_sp(img_path=train_img, label_path=train_label,augment_data=False, target_size=256)
    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)

    if not os.path.exists('labels'):
        os.makedirs('labels')
    
    for ep in range(1,epoch+1):

        model.train()

        #torch.autograd.set_detect_anomaly(True)

        #print(epoch)
        for batch_idx, (inp_imgs, gt_masks, img_names) in enumerate(train_dataloader):

            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            #print('input_img:',inp_imgs.shape)
            optimizer.zero_grad()
            pred_masks= model(inp_imgs)
            
            loss1 = criterion1(pred_masks, gt_masks)
            #print(loss1)
            loss2 = criterion2(pred_masks[:,1,:,:].unsqueeze(1), CRFconfig,inp_imgs)
            loss = torch.mean(loss1)+crf_w*loss2

            
            re = pred_masks[:,1,:,:].detach().cpu().numpy()
            for m in range(pred_masks.shape[0]):
                newlamda = lamda*math.exp(float(ep)/200)
                if loss1[m]<newlamda:
                    #print(new_train_list)
                    new_train_list.append(img_names[m])
                    generate_new_label(pred_masks[m,1,:,:],img_names[m], label_update_dir)



            with torch.autograd.detect_anomaly():
                loss.backward()

            optimizer.step()

            if batch_idx % log_interval == 0:
                k +=1
                print('{} TRAIN :: Epoch : {}\tBatch : {}/{} ({:.2f}%)\t\tTot Loss : {:.4f}'
                        .format(datetime.now(),ep + 1, batch_idx + 1, len(train_dataloader), (batch_idx + 1) * 100 / len(train_dataloader),loss.item()))
                writer.add_scalar('Train/loss', loss.item(), k)
                writer.add_scalar('Train/crfloss', loss2, k)
                writer.add_scalar('Train/lamda',newlamda, k)
                writer.add_image('Train/input', torch.squeeze(inp_imgs[1,:,:,:]), k)
                writer.add_image('Train/output', pred_masks[1,1,:,:].unsqueeze(0), k)
                writer.add_image('Train/gt', gt_masks[1,:,:].unsqueeze(0), k)
                writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], k)

        if ep % 1==0 :
            print('Updating the label..................')
            train_data = dataLoader_sp(img_path=train_img, label_path=label_update_dir,augment_data=False, target_size=256)
            train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
            new_train_list =[]

        # Validation
        if ep % val_interval == 0:
            model.eval()
            val_mae = model_val(model,val_dataloader,criterion3, ep, writer, device, config)

        if ep % 4 == 0 and ep>0:
            torch.save(model.state_dict(), os.path.join(model_path, 'best-model_epoch-{:03}.pth'.
                            format(ep)))

