import torch
import torchvision.models as v_models
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torchvision.ops import roi_pool, RoIPool
from torchsummary import summary
import math
#from data_pre import myDataSet

BATCH_SIZE = 1
R = 10


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def Conv3(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False),nn.ReLU())
def conv3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False)
def Conv1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=1, padding=0, bias=False),nn.ReLU())
def upconv(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, 
                     stride=2, padding=0,output_padding=0, bias= False), nn.ReLU())

class RFA(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(RFA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rb1 = self.RB()
        self.rb2 = self.RB()
        self.conv1 = Conv3(self.in_channels, in_channels)
        self.conv2 = Conv3(3*self.in_channels, 3*self.in_channels)
        self.conv3 = Conv1(3*self.in_channels, self.out_channels)
    
    def RB(self):
        rb_1 = Conv3(self.in_channels, self.in_channels)
        rb_2 = Conv3(self.in_channels, self.in_channels)
        rb = nn.Sequential(rb_1, rb_2)
        return rb

    def forward(self, x):
        c1 = self.rb1(x) #RB1
        c2 = self.rb2(c1 + x) #RB2
        c3 = self.conv1(c2 + c1 + x)
        concat123 = torch.cat((x,c1+x, c3), 1)
        c4 = self.conv2(concat123)
        c5 = self.conv3(c4)
        return c5

class RAN_test(nn.Module):
    def __init__(self):
        super(RAN_test, self).__init__()

        
        self.smooth1 = Conv3(3, 32)
        self.smooth2 = Conv3(32, 64)
        self.smooth3 = Conv3(64, 128)
        self.smooth4 = Conv3(128, 128)
        self.smooth5 = Conv3(128, 64)
        self.smooth6 = Conv3(64, 32)
        self.conv = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        p1 = self.smooth1(x)
        p2 = self.smooth2(p1)
        p3 = self.smooth3(p2)
        p4 = self.smooth4(p3)
        p5 = self.smooth5(p4)
        p6 = self.smooth6(p5)
        
        p7 = self.conv(p6)

        out_logits = p7
        out  = self.softmax(p7)

        # sigma = self.sig_conv3(self.sig_conv2(self.sig_conv1(p1)))
        # out_10 = torch.zeros([10, sigma.size()[0], 1, 256, 256]).cuda()
        # for i in range(10):
        #     epsilon = torch.normal(torch.zeros_like(sigma), 1)
        #     out_10[i,:,:,:,:] = seg_out + sigma*epsilon
        # seg_out = torch.mean(out_10, axis = 0, keepdim = False)
        return out_logits,out




class RAN_un(nn.Module):
    def __init__(self):
        super(RAN_un, self).__init__()

        down_feature =[64, 128, 256, 1024]
        up_feature = 256
        in_channels = 3
        out_channels = 2
        self.in_conv1 = Conv3(in_channels, 32)
        self.relu = nn.ReLU(inplace=True)
        self.in_conv2 = Conv3(32, 32)

        self.maxpool = nn.MaxPool2d(2)
        self.RFA1 = RFA(32, down_feature[0])
        self.RFA2 = RFA(down_feature[0], down_feature[1])
        self.RFA3 = RFA(down_feature[1], down_feature[2])
        self.RFA4 = RFA(down_feature[2], down_feature[3])
        
        self.smooth1 = Conv1(down_feature[2], up_feature)
        self.smooth2 = Conv1(down_feature[1], up_feature)
        self.smooth3 = Conv1(down_feature[0], up_feature)
        self.smooth4 = Conv1(32, up_feature)

        self.conv4_ = Conv3(down_feature[3], up_feature)
        self.deconv4 = upconv(up_feature, up_feature)
        self.conv3_ = Conv3(up_feature, up_feature)
        self.deconv3 = upconv(up_feature, up_feature)
        self.conv2_ = Conv3(up_feature, up_feature)
        self.deconv2 = upconv(up_feature, up_feature)
        self.conv1_ = Conv3(up_feature, up_feature)
        self.deconv1 = upconv(up_feature, up_feature)


        self.out_conv1 = Conv3(up_feature, 64)
        self.out_conv2 = Conv3(64, 32)
        self.out_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.sig_conv1 = Conv3(up_feature, 64)
        self.sig_conv2 = Conv3(64, 32)
        self.sig_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
        #self.dropout = nn.Dropout(p=0.1)
        #self.dropout = nn.Dropout2d(p=0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        c1 = self.in_conv2(self.in_conv1(x))
        
        c2 = self.RFA1(self.maxpool(c1))
        c3 = self.RFA2(self.maxpool(c2))
        c4 = self.RFA3(self.maxpool(c3))
        c5 = self.RFA4(self.maxpool(c4))

        p5 = self.conv4_(c5)
        p4 = self.smooth1(c4)+self.deconv4(p5)
        p3 = self.smooth2(c3)+self.deconv3(self.conv3_(p4))
        p2 = self.smooth3(c2)+self.deconv2(self.conv2_(p3))
        p1 = self.smooth4(c1)+self.deconv1(self.conv1_(p2))

        logits = self.out_conv3(self.out_conv2(self.out_conv1(p1)))
        #logits = self.out_conv3(self.out_conv2(self.out_conv1(p1)))
        #out  = self.softmax(out_logits)

        logvar = self.sig_conv3(self.sig_conv2(self.sig_conv1(p1)))
        logvar = torch.exp(logvar/2)
        # print(logits.size())
        epsilon = torch.randn_like(logvar)
        prev_attn = epsilon * logvar
        out_logits = logits + prev_attn
        seg_out = self.softmax(out_logits)
        #out = self.softmax(logits)

        return seg_out, logvar

class RAN(nn.Module):
    def __init__(self):
        super(RAN, self).__init__()

        down_feature = [64, 128, 256, 1024]
        up_feature = 128
        in_channels = 3
        out_channels = 2
        self.embedding=[]
        self.in_conv1 = Conv3(in_channels, 32)
        self.relu = nn.ReLU(inplace=True)
        self.in_conv2 = Conv3(32, 32)

        self.maxpool = nn.MaxPool2d(2)
        self.RFA1 = RFA(32, down_feature[0])
        self.RFA2 = RFA(down_feature[0], down_feature[1])
        self.RFA3 = RFA(down_feature[1], down_feature[2])
        self.RFA4 = RFA(down_feature[2], down_feature[3])
        
        self.smooth1 = Conv1(down_feature[2], up_feature)
        self.smooth2 = Conv1(down_feature[1], up_feature)
        self.smooth3 = Conv1(down_feature[0], up_feature)
        self.smooth4 = Conv1(32, up_feature)

        self.conv4_ = Conv3(down_feature[3], up_feature)
        self.deconv4 = upconv(up_feature, up_feature)
        self.conv3_ = Conv3(up_feature, up_feature)
        self.deconv3 = upconv(up_feature, up_feature)
        self.conv2_ = Conv3(up_feature, up_feature)
        self.deconv2 = upconv(up_feature, up_feature)
        self.conv1_ = Conv3(up_feature, up_feature)
        self.deconv1 = upconv(up_feature, up_feature)


        self.out_conv1 = Conv3(up_feature, 64)
        self.out_conv2 = Conv3(64, 32)
        self.out_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.sig_conv1 = Conv3(up_feature, 64)
        self.sig_conv2 = Conv3(64, 32)
        self.sig_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        c1 = self.in_conv2(self.in_conv1(x))
        
        c2 = self.RFA1(self.maxpool(c1))
        c3 = self.RFA2(self.maxpool(c2))
        c4 = self.RFA3(self.maxpool(c3))
        c5 = self.RFA4(self.maxpool(c4))

        p5 = self.conv4_(c5)
        p4 = self.smooth1(c4)+self.deconv4(p5)
        p3 = self.smooth2(c3)+self.deconv3(self.conv3_(p4))
        p2 = self.smooth3(c2)+self.deconv2(self.conv2_(p3))
        p1 = self.smooth4(c1)+self.deconv1(self.conv1_(p2))

        self.embedding = [p1,p2,p3,p4,p5]
        logits = self.out_conv3(self.out_conv2(self.out_conv1(p1)))
        out  = self.softmax(logits)

    

        return out

#0145
# class RAN_un_com(nn.Module):
#     def __init__(self):
#         super(RAN_un_com, self).__init__()

#         down_feature = [64, 128, 256, 1024]
#         up_feature = 256
#         in_channels = 3
#         out_channels = 2
#         self.in_conv1 = Conv3(in_channels, 32)
#         self.relu = nn.ReLU(inplace=True)
#         self.in_conv2 = Conv3(32, 32)

#         self.maxpool = nn.MaxPool2d(2)
#         self.RFA1 = RFA(32, down_feature[0])
#         self.RFA2 = RFA(down_feature[0], down_feature[1])
#         self.RFA3 = RFA(down_feature[1], down_feature[2])
#         self.RFA4 = RFA(down_feature[2], down_feature[3])
        
#         self.smooth1 = Conv1(down_feature[2], up_feature)
#         self.smooth2 = Conv1(down_feature[1], up_feature)
#         self.smooth3 = Conv1(down_feature[0], up_feature)
#         self.smooth4 = Conv1(32, up_feature)

#         self.conv4_ = Conv3(down_feature[3], up_feature)
#         self.deconv4 = upconv(up_feature, up_feature)
#         self.conv3_ = Conv3(up_feature, up_feature)
#         self.deconv3 = upconv(up_feature, up_feature)
#         self.conv2_ = Conv3(up_feature, up_feature)
#         self.deconv2 = upconv(up_feature, up_feature)
#         self.conv1_ = Conv3(up_feature, up_feature)
#         self.deconv1 = upconv(up_feature, up_feature)


#         self.out_conv1 = Conv3(up_feature, 64)
#         self.out_conv2 = Conv3(64, 32)
#         self.out_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)

#         self.sig_conv1 = Conv3(up_feature, 64)
#         self.sig_conv2 = Conv3(64, 32)
#         self.sig_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
#         self.dropout1 = nn.Dropout2d(p=0.5)
#         self.dropout2 = nn.Dropout2d(p=0.5)
#         self.dropout3 = nn.Dropout2d(p=0.5)
#         self.dropout4 = nn.Dropout2d(p=0.5)
#         self.dropout5 = nn.Dropout2d(p=0.5)
#         self.dropout6 = nn.Dropout2d(p=0.5)
#         self.dropout7 = nn.Dropout2d(p=0.5)
#         self.dropout8 = nn.Dropout2d(p=0.5)
#         self.dropout9 = nn.Dropout2d(p=0.5)
#         self.dropout10 = nn.Dropout2d(p=0.5)
#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()


#     def forward(self, x):

#         c1 = self.in_conv2(self.in_conv1(x))
        
#         c2 = self.RFA1(self.maxpool(c1))
#         c3 = self.RFA2(self.maxpool(c2))
#         c4 = self.RFA3(self.maxpool(c3))
#         c5 = self.RFA4(self.maxpool(c4))

#         p5 = self.dropout1(self.conv4_(c5))
#         p4 = self.dropout2(self.smooth1(c4)+self.deconv4(p5))
#         p3 = self.dropout3(self.smooth2(c3)+self.deconv3(self.conv3_(p4)))
#         p2 = self.dropout4(self.smooth3(c2)+self.deconv2(self.conv2_(p3)))
#         p1 = self.dropout5(self.smooth4(c1)+self.deconv1(self.conv1_(p2)))
#         logits = self.out_conv3(self.dropout7(self.out_conv2(self.dropout6(self.out_conv1(p1)))))


#         logvar = self.sig_conv3(self.dropout9(self.sig_conv2(self.dropout8(self.sig_conv1(p1)))))
#         logvar = torch.exp(logvar/2)
#         epsilon = torch.randn_like(logvar)
#         prev_attn = epsilon * logvar
#         out_logits = logits + prev_attn
#         seg_out = self.softmax(out_logits)

#         return seg_out, logvar

#1823 
# class RAN_un_com(nn.Module):
#     def __init__(self):
#         super(RAN_un_com, self).__init__()

#         down_feature = [64, 128, 256, 1024]
#         up_feature = 256
#         in_channels = 3
#         out_channels = 2
#         self.in_conv1 = Conv3(in_channels, 32)
#         self.relu = nn.ReLU(inplace=True)
#         self.in_conv2 = Conv3(32, 32)

#         self.maxpool = nn.MaxPool2d(2)
#         self.RFA1 = RFA(32, down_feature[0])
#         self.RFA2 = RFA(down_feature[0], down_feature[1])
#         self.RFA3 = RFA(down_feature[1], down_feature[2])
#         self.RFA4 = RFA(down_feature[2], down_feature[3])
        
#         self.smooth1 = Conv1(down_feature[2], up_feature)
#         self.smooth2 = Conv1(down_feature[1], up_feature)
#         self.smooth3 = Conv1(down_feature[0], up_feature)
#         self.smooth4 = Conv1(32, up_feature)

#         self.conv4_ = Conv3(down_feature[3], up_feature)
#         self.deconv4 = upconv(up_feature, up_feature)
#         self.conv3_ = Conv3(up_feature, up_feature)
#         self.deconv3 = upconv(up_feature, up_feature)
#         self.conv2_ = Conv3(up_feature, up_feature)
#         self.deconv2 = upconv(up_feature, up_feature)
#         self.conv1_ = Conv3(up_feature, up_feature)
#         self.deconv1 = upconv(up_feature, up_feature)


#         self.out_conv1 = Conv3(up_feature, 64)
#         self.out_conv2 = Conv3(64, 32)
#         self.out_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)

#         self.sig_conv1 = Conv3(up_feature, 64)
#         self.sig_conv2 = Conv3(64, 32)
#         self.sig_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
#         self.dropout1 = nn.Dropout2d(p=0.2)
#         self.dropout2 = nn.Dropout2d(p=0.3)
#         self.dropout3 = nn.Dropout2d(p=0.3)
#         self.dropout4 = nn.Dropout2d(p=0.3)
#         self.dropout5 = nn.Dropout2d(p=0.3)
#         self.dropout6 = nn.Dropout2d(p=0.1)
#         self.dropout7 = nn.Dropout2d(p=0.1)
#         self.dropout8 = nn.Dropout2d(p=0.3)
#         self.dropout9 = nn.Dropout2d(p=0.2)
#         self.dropout10 = nn.Dropout2d(p=0.2)
#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()


#     def forward(self, x):

#         c1 = self.in_conv2(self.in_conv1(x))
        
#         c2 = self.RFA1(self.maxpool(c1))
#         c3 = self.RFA2(self.maxpool(c2))
#         c4 = self.dropout9(self.RFA3(self.maxpool(c3)))
#         c5 = self.dropout10(self.RFA4(self.maxpool(c4)))

#         p5 = self.dropout1(self.conv4_(c5))
#         p4 = self.dropout2(self.smooth1(c4)+self.deconv4(p5))
#         p3 = self.dropout3(self.smooth2(c3)+self.deconv3(self.conv3_(p4)))
#         p2 = self.dropout4(self.smooth3(c2)+self.deconv2(self.conv2_(p3)))
#         p1 = self.dropout5(self.smooth4(c1)+self.deconv1(self.conv1_(p2)))
#         logits = self.out_conv3(self.dropout7((self.out_conv2(self.dropout6(self.out_conv1(p1))))))


#         logvar = self.sig_conv3(self.sig_conv2(self.sig_conv1(p1)))
#         logvar = torch.exp(logvar/2)
#         epsilon = torch.randn_like(logvar)
#         prev_attn = epsilon * logvar
#         out_logits = logits + prev_attn
#         seg_out = self.softmax(out_logits)

#         return seg_out, logvar

class RAN_un_com(nn.Module):
    def __init__(self):
        super(RAN_un_com, self).__init__()

        down_feature = [64, 128, 256, 1024]
        up_feature = 256
        in_channels = 3
        out_channels = 2
        self.in_conv1 = Conv3(in_channels, 32)
        self.relu = nn.ReLU(inplace=True)
        self.in_conv2 = Conv3(32, 32)

        self.maxpool = nn.MaxPool2d(2)
        self.RFA1 = RFA(32, down_feature[0])
        self.RFA2 = RFA(down_feature[0], down_feature[1])
        self.RFA3 = RFA(down_feature[1], down_feature[2])
        self.RFA4 = RFA(down_feature[2], down_feature[3])
        
        self.smooth1 = Conv1(down_feature[2], up_feature)
        self.smooth2 = Conv1(down_feature[1], up_feature)
        self.smooth3 = Conv1(down_feature[0], up_feature)
        self.smooth4 = Conv1(32, up_feature)

        self.conv4_ = Conv3(down_feature[3], up_feature)
        self.deconv4 = upconv(up_feature, up_feature)
        self.conv3_ = Conv3(up_feature, up_feature)
        self.deconv3 = upconv(up_feature, up_feature)
        self.conv2_ = Conv3(up_feature, up_feature)
        self.deconv2 = upconv(up_feature, up_feature)
        self.conv1_ = Conv3(up_feature, up_feature)
        self.deconv1 = upconv(up_feature, up_feature)


        self.out_conv1 = Conv3(up_feature, 64)
        self.out_conv2 = Conv3(64, 32)
        self.out_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.sig_conv1 = Conv3(up_feature, 64)
        self.sig_conv2 = Conv3(64, 32)
        self.sig_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.dropout2 = nn.Dropout2d(p=0.3)
        self.dropout3 = nn.Dropout2d(p=0.3)
        self.dropout4 = nn.Dropout2d(p=0.3)
        self.dropout5 = nn.Dropout2d(p=0.3)
        self.dropout6 = nn.Dropout2d(p=0.1)
        self.dropout7 = nn.Dropout2d(p=0.1)
        self.dropout8 = nn.Dropout2d(p=0.3)
        self.dropout9 = nn.Dropout2d(p=0.3)
        self.dropout10 = nn.Dropout2d(p=0.3)
        self.dropout11 = nn.Dropout2d(p=0.3)
        self.dropout12 = nn.Dropout2d(p=0.3)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        c1 = self.in_conv2(self.in_conv1(x))
        
        c2 = self.dropout11(self.RFA1(self.maxpool(c1)))
        c3 = self.dropout12(self.RFA2(self.maxpool(c2)))
        c4 = self.dropout9(self.RFA3(self.maxpool(c3)))
        c5 = self.dropout10(self.RFA4(self.maxpool(c4)))

        p5 = self.dropout1(self.conv4_(c5))
        p4 = self.dropout2(self.smooth1(c4)+self.deconv4(p5))
        p3 = self.dropout3(self.smooth2(c3)+self.deconv3(self.conv3_(p4)))
        p2 = self.dropout4(self.smooth3(c2)+self.deconv2(self.conv2_(p3)))
        p1 = self.dropout5(self.smooth4(c1)+self.deconv1(self.conv1_(p2)))
        logits = self.out_conv3(self.dropout7((self.out_conv2(self.dropout6(self.out_conv1(p1))))))


        logvar = self.sig_conv3(self.sig_conv2(self.sig_conv1(p1)))
        logvar = torch.exp(logvar/2)
        epsilon = torch.randn_like(logvar)
        prev_attn = epsilon * logvar
        out_logits = logits + prev_attn
        seg_out = self.softmax(out_logits)

        return seg_out, logvar


class Self_Attention(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Self_Attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv3(self.in_channels, self.out_channels)
        self.conv2 = conv3(self.in_channels, self.out_channels)
        self.conv3 = Conv3(self.in_channels, self.out_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)        
        


    def forward(self, x_o):
        self.interpolation = nn.UpsamplingBilinear2d(size=x_o.shape[2:4])
        x = self.avgpool(x_o)
        b,h,w =x.shape[0], x.shape[2],x.shape[3]
        B_1, B_2, B_3 = self.conv1(x),self.conv2(x),self.conv3(x)
        B_1,B_2, B_3 = B_1.view(-1,h*w,self.out_channels), B_2.view(-1,self.out_channels,h*w), B_3.view(-1,h*w,self.out_channels)
        #print(B_1.shape,B_2.shape,B_3.shape)
        self.attention = F.softmax(torch.matmul(B_1,B_2)/math.sqrt(b*h*w), dim=-1)
        #print(self.attention.shape)
        self.feature = torch.matmul(self.attention, B_3)
        self.feature = self.feature.view(-1,self.out_channels, h,w)
        self.feature = self.interpolation(self.feature)
        #print(self.feature.shape)

        return self.feature

class Att_RAN(nn.Module):
    def __init__(self):
        super(Att_RAN, self).__init__()

        down_feature =[128, 256, 512, 1024]
        up_feature = 256
        in_channels = 3
        out_channels = 2
        self.in_features = []
        self.in_conv1 = Conv3(in_channels, 32)
        self.relu = nn.ReLU(inplace=True)
        self.in_conv2 = Conv3(32, 32)

        self.maxpool = nn.MaxPool2d(2)
        self.RFA1 = RFA(32, down_feature[0])
        self.RFA2 = RFA(down_feature[0], down_feature[1])
        self.RFA3 = RFA(down_feature[1], down_feature[2])
        self.RFA4 = RFA(down_feature[2], down_feature[3])

        self.smooth1 = Self_Attention(down_feature[2], up_feature)
        self.smooth1_1 = Conv3(up_feature,up_feature)
        self.smooth2 = Self_Attention(down_feature[1], up_feature)
        self.smooth2_1 = Conv3(up_feature,up_feature)
        self.smooth3 = Self_Attention(down_feature[0], up_feature)
        self.smooth3_1 = Conv3(up_feature,up_feature)
        self.smooth4 = Conv1(32, up_feature)
                
        self.conv4_ = Conv3(down_feature[3], up_feature)
        self.deconv4 = upconv(up_feature, up_feature)
        self.conv3_ = Conv3(up_feature, up_feature)
        self.deconv3 = upconv(up_feature, up_feature)
        self.conv2_ = Conv3(up_feature, up_feature)
        self.deconv2 = upconv(up_feature, up_feature)
        self.conv1_ = Conv3(up_feature, up_feature)
        self.deconv1 = upconv(up_feature, up_feature)
        

        self.out_conv1 = Conv3(up_feature, 64)
        self.out_conv2 = Conv3(64, 32)
        self.out_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
    
        self.softmax = nn.Softmax(dim=1)
        self.encoder_feature = []



    def forward(self, x):

        c1 = self.in_conv2(self.in_conv1(x))
        
        c2 = self.RFA1(self.maxpool(c1))
        c3 = self.RFA2(self.maxpool(c2))
        c4 = self.RFA3(self.maxpool(c3))
        c5 = self.RFA4(self.maxpool(c4))
        p5 = self.conv4_(c5)
        p4 = self.smooth1_1(self.smooth1(c4))+self.deconv4(p5)
        p3 = self.smooth2_1(self.smooth2(c3))+self.deconv3(self.conv3_(p4))
        p2 = self.smooth3_1(self.smooth3(c2))+self.deconv2(self.conv2_(p3))
        p1 = self.smooth4(c1)+self.deconv1(self.conv1_(p2))

        self.logits = self.out_conv3(self.out_conv2(self.out_conv1(p1)))
        self.out  = self.softmax(self.logits)

        return self.out


class Att_Acc_RAN(nn.Module):
    def __init__(self):
        super(Att_Acc_RAN, self).__init__()

        down_feature =[128, 256, 512, 256]
        self.embedding=[]
        up_feature = 256
        in_channels = 3
        out_channels = 2
        self.in_features = []
        self.in_conv1 = Conv3(in_channels, 32)
        self.relu = nn.ReLU(inplace=True)
        self.in_conv2 = Conv3(32, 32)

        self.maxpool = nn.MaxPool2d(2)
        self.RFA1 = RFA(32, down_feature[0])
        self.RFA2 = RFA(down_feature[0], down_feature[1])
        self.RFA3 = RFA(down_feature[1], down_feature[2])
        self.RFA4 = RFA(down_feature[2], down_feature[3])

        self.smooth0 =  Conv3(up_feature, 32)
        self.smooth1 = Self_Attention(up_feature, up_feature)
        self.smooth1_1 = Conv3(up_feature,up_feature)
        self.smooth2 = Self_Attention(up_feature, up_feature)
        self.smooth2_1 = Conv3(up_feature,up_feature)
        self.smooth3 = Self_Attention(up_feature, up_feature)
        self.smooth3_1 = Conv3(up_feature,up_feature)
        self.smooth4 =  Self_Attention(up_feature,up_feature)
        self.smooth4_1 =Conv3(up_feature, up_feature)

        self.conv4_ = Conv3(down_feature[3], up_feature)
        self.deconv4 = upconv(up_feature, up_feature)
        self.conv3_ = Conv3(down_feature[2], up_feature)
        self.deconv3 = upconv(up_feature, up_feature)
        self.conv2_ = Conv3(down_feature[1], up_feature)
        self.deconv2 = upconv(up_feature, up_feature)
        self.conv1_ = Conv3(down_feature[0], up_feature)
        self.deconv1 = upconv(32, 32)
        self.conv0_ = Conv3(32, up_feature)

        self.conv4_1 = Conv3(up_feature, up_feature)
        self.conv3_1 = Conv3(up_feature, up_feature)
        self.conv2_1 = Conv3(up_feature, up_feature)
        self.conv1_1 = Conv3(up_feature, 32)



        self.upp_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upp_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upp_3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upp_4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upp1_1 = Conv3(up_feature, up_feature)
        self.upp2_1 = Conv3(up_feature, up_feature)
        self.upp3_1 = Conv3(up_feature, up_feature)
        self.upp4_1 = Conv3(up_feature, up_feature)

        self.weights = nn.Parameter(torch.ones(4), requires_grad= True)

        self.out_conv1 = Conv3(32, 32)
        self.out_conv2 = Conv3(32, 16)
        self.out_conv3 = nn.Conv2d(16, 2, kernel_size=1, stride=1, padding=0, bias=False)
    
        self.softmax = nn.Softmax(dim=1)
        self.encoder_feature = []

    def Att_Acc(self, Att_F, f):
        #print('Att_F:',Att_F.shape)
        #print('f:',f.shape)
        C,H,W = Att_F.shape[1],Att_F.shape[2],Att_F.shape[3]
        #print(C,H,W)

        Att_F_1, f_1 = Att_F.view(-1,H*W,C), f.view(-1,H*W,C)
        f_2 = torch.sum(f_1,dim=-2, keepdim=True)/(H*W)
        #print('Att_F_1:',Att_F_1.shape)
        #print('f_1:',f_2.shape)

        Projected_feature = torch.sum(Att_F_1*f_2,dim=-1,keepdim=True)/(torch.sqrt(torch.sum(Att_F_1**2,dim=-1,keepdim=True)+1e-8)*torch.sqrt(torch.sum(f_2**2,dim=-1,keepdim=True)+1e-8)+ 1e-5)*Att_F_1
        Orth_feature = Att_F_1-Projected_feature
        # print('ATT:',Att_F_1)
        # print('f_2:',f_2)
        # print('orth:',Orth_feature)
        return Orth_feature.view(-1,C, H,W)

    def forward(self, x):

        c1 = self.in_conv2(self.in_conv1(x))
        
        c2 = self.RFA1(self.maxpool(c1))
        c3 = self.RFA2(self.maxpool(c2))
        c4 = self.RFA3(self.maxpool(c3))
        c5 = self.RFA4(self.maxpool(c4))
        #print('c:', c1.shape,c2.shape,c3.shape,c4.shape,c5.shape)
        cc5 = self.conv4_(c5)
        cc4 = self.conv3_(c4)
        cc3 = self.conv2_(c3)
        cc2 = self.conv1_(c2)
        cc1 = self.conv0_(c1)
        #print('cc:', cc1.shape,cc2.shape,cc3.shape,cc4.shape,cc5.shape)

        p5 = self.smooth4(cc5)
        a5 = self.Att_Acc(p5,cc5)
        upp_5 = self.upp_4(a5)
        # print('p5:', p5.shape)
        # print('a5:', a5.shape)
        # print('upp_5:', upp_5.shape)

        a4 = self.smooth3(self.weights[3]*upp_5+cc4)
        p4 = a4 +self.deconv4(self.conv4_1(p5))
        upp_4 =  self.upp_3(self.Att_Acc(p4,cc4))

        # print('p4:', p4.shape)
        # print('a4:', a4.shape)
        # print('upp_4:', upp_4.shape)

        a3 = self.smooth2(self.weights[2]*upp_4+cc3)
        p3 = a3 +self.deconv3(self.conv3_1(p4))
        upp_3 = self.upp_2(self.Att_Acc(p3,cc3))
        # print('p3:', p3.shape)
        # print('a3:', a3.shape)
        # print('upp_3:', upp_3.shape)

        a2 = self.smooth1(self.weights[1]*upp_3+ cc2)
        p2 = a2 +self.deconv2(self.conv2_1(p3))
        upp_2 = self.upp_1(self.Att_Acc(p2,cc2))
        # print('p2:', p2.shape)
        # print('a2:', a2.shape)
        # print('upp_2:', upp_2.shape)

        a1 = self.smooth0(self.weights[0]*upp_2+ cc1)
        p1 = a1 +self.deconv1(self.conv1_1(p2))
        # print('p1:', p1.shape)
        # print('a1:', a1.shape)
        self.embedding = [a1,a2,a3,a4,a5]
        self.logits = self.out_conv3(self.out_conv2(self.out_conv1(p1)))
        self.out  = self.softmax(self.logits)

        return self.out