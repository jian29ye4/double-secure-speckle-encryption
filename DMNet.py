from math import pi
import time
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexDropout
from complexFunctions import complex_relu, complex_max_pool2d

class DoubleKey5(nn.Module):
    '''
    optical propagation:
    image1 + image2 = speckles

    image1 is QR code
    image2 are from fMNIST.

        (input1)x1:qr       (input2)x2:full-speckle
            |                       |
    f1(qr):qr_speckle               |
            |                       |
            ------------|------------            
                        |
            x_diff = f_diff(f1(input1),input2): (full-speckle-qr_speckle=image_speckle)
                        |
                y = f_out(x_diff): image generated from image_speckle

    f1:     Complex_fc
    f_diff: Complex_Unet
    f_out:  Comlex_fc
    '''
    def __init__(self, in1_dim, in2_dim, out_dim, dropout):
        super(DoubleKey5, self).__init__()
        self.in1_dim = in1_dim
        self.in2_dim = in2_dim
        self.out_dim = out_dim
        self.p = dropout
        self.fc_qr2spk = ComplexLinear(self.in1_dim*self.in1_dim, self.in2_dim*self.in2_dim) #f1
        self.fc_out = ComplexLinear(self.in2_dim*self.in2_dim, self.out_dim*self.out_dim) #f2
        self.norm1 = SwitchNorm2d(1)#   nn.InstanceNorm2d(1, affine=True)#nn.InstanceNorm2d(self.in2_dim*self.in2_dim)
        self.norm2 = SwitchNorm2d(1)#nn.InstanceNorm2d(1, affine=True)#nn.LayerNorm(self.in2_dim*self.in2_dim)
        self.norm3 = nn.LayerNorm(self.out_dim*self.out_dim)
        self.unet = Complex_Unet(dropout=self.p, inputCH=2)
        self.dropout1 = ComplexDropout(p=self.p)
        self.dropout2 = ComplexDropout(p=self.p)
        self.dp = nn.Dropout2d(p=self.p) #fine-tune : p=0.8
        self.sigmoid = nn.Sigmoid()
        self.bn = SwitchNorm2d(1)#nn.BatchNorm2d(1)
        #self.res = resnext101() #f_diff

        
    def forward(self, x1, x2):
        #x1: QRCODE; x2:speckle
        

        # input1 into f1: qr->qr_speckle
        x1 = x1.view(-1, self.in1_dim*self.in1_dim)
        x1_im = torch.zeros(x1.shape, dtype = x1.dtype, device = x1.device)
        x1, x1_im = self.fc_qr2spk(x1, x1_im)
        x1, x1_im = self.dropout1(x1, x1_im)
        x1 =  torch.sqrt(torch.pow(x1,2)+torch.pow(x1_im,2))
        x1 = self.norm1(x1.view(-1, 1, self.in2_dim, self.in2_dim))
        x2 = self.norm2(torch.sqrt(x2))
        f_diff_r = self.unet(torch.cat([x1,x2],dim=1));f_diff_r = torch.sqrt(f_diff_r)
        f_diff_r = f_diff_r.view(-1, self.in2_dim*self.in2_dim);f_diff_i = torch.zeros(f_diff_r.shape, dtype = f_diff_r.dtype, device = f_diff_r.device)
        
      # f1(input1), input2 into f_diff
        f_diff_r, f_diff_i = self.fc_out(f_diff_r, f_diff_i)
        f_diff_r, f_diff_i = self.dropout2(f_diff_r, f_diff_i)
        x = torch.sqrt(torch.pow(f_diff_r,2)+torch.pow(f_diff_i,2))
        x = self.bn(x.view(-1, self.out_dim, self.out_dim).unsqueeze(1))

        return x

class Complex_Unet(torch.nn.Module):
    def __init__(self, dropout=0.5, GAN_type=1, inputCH=2):
        class Real_to_Complex(torch.nn.Module): # Transform Real number to Complex number
            def forward(self, input): 
                if GAN_type == 1: return input, torch.zeros(input.shape, dtype=input.dtype, device=input.device)
                if GAN_type == 0: return input
        
        class Complex_to_Real(torch.nn.Module): # Transform Complex number to Real number
            def forward(self, input):
                if GAN_type == 1: return torch.sqrt(input[0].square() + input[1].square())
                if GAN_type == 0: return input

        class Complex_conv2d(torch.nn.Module): # Complex Conv2d, perform Conv2d independently on real and imaginary part.
            def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
                super(Complex_conv2d, self).__init__()
                if GAN_type == 1: self.conv_r, self.conv_i = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
                if GAN_type == 0: self.conv_r              = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            def forward(self, input): 
                if GAN_type == 1: return self.conv_r(input[0])-self.conv_i(input[1]), self.conv_r(input[0])+self.conv_i(input[1])
                if GAN_type == 0: return self.conv_r(input)

        class Complex_ConvTranspose2d(torch.nn.Module):
            def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
                super(Complex_ConvTranspose2d, self).__init__()
                if GAN_type == 1: self.conv_tran_r, self.conv_tran_i = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode), torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
                if GAN_type == 0: self.conv_tran_r                   = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
            def forward(self, input):
                if GAN_type == 1: return self.conv_tran_r(input[0])-self.conv_tran_i(input[1]), self.conv_tran_r(input[0])+self.conv_tran_i(input[1])
                if GAN_type == 0: return self.conv_tran_r(input)

        class Complex_ReLU(torch.nn.Module): # Complex ReLU, perform ReLU independently on real and imaginary part.
            def __init__(self, inplace=False):
                super(Complex_ReLU, self).__init__()
                if GAN_type == 1: self.ReLU_r, self.ReLU_i = torch.nn.ReLU(inplace),torch.nn.ReLU(inplace)
                if GAN_type == 0: self.ReLU_r              = torch.nn.ReLU(inplace)
            def forward(self, input):
                if GAN_type == 1: return self.ReLU_r(input[0]), self.ReLU_i(input[1])
                if GAN_type == 0: return self.ReLU_r(input)

        class Complex_batch_norm2d(torch.nn.Module): # Complex BatchNorm, perform BatchNorm independently on real and imaginary part.
            def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
                super(Complex_batch_norm2d, self).__init__()
                if GAN_type == 1: self.bn_r, self.bn_i = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats),torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
                if GAN_type == 0: self.bn_r            = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
            def forward(self, input): 
                if GAN_type == 1: return self.bn_r(input[0]), self.bn_i(input[1])
                if GAN_type == 0: return self.bn_r(input)

        class Complex_dropout2d(torch.nn.Module): # Complex Dropout, perform Dropout independently on real and imaginary part.
            def __init__(self,p=dropout, inplace=False):
                super(Complex_dropout2d,self).__init__()
                if GAN_type == 1: self.dropout_r, self.dropout_i = torch.nn.Dropout2d(p,inplace), torch.nn.Dropout2d(p,inplace)
                if GAN_type == 0: self.dropout_r                 = torch.nn.Dropout2d(p,inplace)
            def forward(self, input):
                if GAN_type == 1: return self.dropout_r(input[0]), self.dropout_i(input[1])
                if GAN_type == 0: return self.dropout_r(input)

        class Complex_Sigmoid(torch.nn.Module):
            def __init__(self, inplace=False):
                super(Complex_Sigmoid,self).__init__()
                if GAN_type == 1: self.sigmoid_r, self.sigmoid_i = torch.nn.Sigmoid(), torch.nn.Sigmoid()
                if GAN_type == 0: self.sigmoid_r                 = torch.nn.Sigmoid()
            def forward(self, input):
                if GAN_type == 1: return self.sigmoid_r(input[0]), self.sigmoid_i(input[1])
                if GAN_type == 0: return self.sigmoid_r(input)
        
        class hook(torch.nn.Module):
            def __init__(self): super(hook,self).__init__()
            def forward(self, input):
                if GAN_type == 1: print(f'hook r:{input[0].shape} hook i:{input[1].shape}')
                if GAN_type == 0: print(f'hook r:{input.shape   } hook i:N/A')
                return input

        class DenseBlock(torch.nn.Module): # DenseLayer1 -> DenseLayer2 -> ... -> Conv2d -> BatchNorm2d -> ReLU
            def __init__(self, in_ch, out_ch, growth_rate=16, num_layer=4, drop_rate=0):
                super(DenseBlock, self).__init__()
                class DenseLayer(torch.nn.Module): # Conv2d -> BatchNorm2d -> ReLU -> Dropout2d
                    def __init__(self, in_ch, out_ch, drop_rate=dropout):
                        super(DenseLayer, self).__init__()
                        self.conv = torch.nn.Sequential(Complex_conv2d(in_ch, out_ch, 3, 1, 1), Complex_batch_norm2d(out_ch), Complex_ReLU(), Complex_dropout2d(drop_rate))
                    def forward(self, input):  return self.conv(input)
                #if opt.dense_layer_drop != 0: 
                drop_rate = 0.5 # DenseLayer using drop_rate=0.5 may have better performance
                self.DenseLayer1 = DenseLayer(in_ch=in_ch + 0,             out_ch=growth_rate, drop_rate=drop_rate )
                self.DenseLayer2 = DenseLayer(in_ch=in_ch + growth_rate,   out_ch=growth_rate, drop_rate=drop_rate )
                self.DenseLayer3 = DenseLayer(in_ch=in_ch + growth_rate*2, out_ch=growth_rate, drop_rate=drop_rate )
                self.DenseLayer4 = DenseLayer(in_ch=in_ch + growth_rate*3, out_ch=out_ch,      drop_rate=drop_rate )
            def forward(self, x):
                if GAN_type == 1: 
                    d_r, d_i = [x[0]],[x[1]]
                    tmp = self.DenseLayer1 (x)
                    d_r.append(tmp[0]), d_i.append(tmp[1])
                    tmp = self.DenseLayer2 ((torch.cat(d_r, 1),torch.cat(d_i, 1)))
                    d_r.append(tmp[0]), d_i.append(tmp[1])
                    tmp = self.DenseLayer3 ((torch.cat(d_r, 1),torch.cat(d_i, 1)))
                    d_r.append(tmp[0]), d_i.append(tmp[1])
                    return self.DenseLayer4((torch.cat(d_r, 1),torch.cat(d_i, 1)))
                if GAN_type == 0: 
                    d = [x]
                    d.append(self.DenseLayer1(x))
                    d.append(self.DenseLayer2(torch.cat(d, 1)))
                    d.append(self.DenseLayer3(torch.cat(d, 1)))
                    return   self.DenseLayer4(torch.cat(d, 1))
                                
        class down(torch.nn.Module): #DenseBlock -> Dropout2d -> Conv2d -> BatchNorm2d -> ReLU -> Dropout2d
            def __init__(self, dropout=0, in_ch=1, out_ch=1, kernel_size=4, padding=1, stride=2):
                super(down, self).__init__()
                self.same = torch.nn.Sequential(DenseBlock(out_ch=out_ch, in_ch=in_ch, drop_rate=dropout), Complex_dropout2d(dropout))
                self.d = torch.nn.Sequential(Complex_conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride), Complex_batch_norm2d(out_ch), Complex_ReLU(), Complex_dropout2d(dropout))
            def forward(self, x):
                x_skip = self.same(x)
                return x_skip, self.d(x_skip)

        class up(torch.nn.Module): # ConvTranspose2d -> BatchNorm2d -> ReLU -> DenseBlock -> Dropout2d
            def __init__(self, dropout=0, in_ch=1, in_m_ch=1, out_ch=1, kernel_size=4, padding=1, stride=2):
                super(up, self).__init__()
                self.u = torch.nn.Sequential(Complex_ConvTranspose2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, stride=stride), Complex_batch_norm2d(in_ch), Complex_ReLU())
                self.h = torch.nn.Sequential(DenseBlock(out_ch=out_ch, in_ch=in_m_ch, drop_rate=dropout), Complex_dropout2d(dropout))
            if GAN_type == 1:
                def forward(self, x, x_skip):
                    x = self.u(x)
                    x = torch.cat([x[0],x_skip[0]],1), torch.cat([x[1],x_skip[1]],1)
                    return self.h(x)
            if GAN_type == 0:
                def forward(self, x, x_skip): return self.h(torch.cat([self.u(x), x_skip], 1))

        class top_out(torch.nn.Module): # Conv2d -> BatchNorm2d -> ReLU -> Dropout2d -> Conv2d -> Sigmoid
            def __init__(self, dropout=0, in_ch=1, kernel_size=3, stride=1, padding=1):
                super(top_out, self).__init__()
                self.cnn = torch.nn.Sequential(Complex_conv2d(in_ch, 128, kernel_size=kernel_size, stride=stride, padding=padding), Complex_batch_norm2d(128),
                Complex_ReLU(), Complex_dropout2d(dropout), Complex_conv2d(128, 1, kernel_size=kernel_size, stride=stride, padding=padding), Complex_Sigmoid())
            def forward(self, x): return self.cnn(x)

        super(Complex_Unet, self).__init__()
        self.channel_number  = 64
        self.Real_to_Complex = Real_to_Complex()
        self.inputCH = inputCH
        self.s1_down = down(dropout, in_ch=self.inputCH, out_ch=self.channel_number,    kernel_size=4, padding=1, stride=2)#  64x256x256
        #in_ch=2 for 2to1 mapping, in_ch=1 for 1to1 mapping
        self.s2_down = down(dropout, in_ch=self.channel_number,   out_ch=self.channel_number*2,  kernel_size=4, padding=1, stride=2)# 128x128x128
        self.s3_down = down(dropout, in_ch=self.channel_number*2, out_ch=self.channel_number*4,  kernel_size=4, padding=1, stride=2)# 256x 64x 64
        self.s4_down = down(dropout, in_ch=self.channel_number*4, out_ch=self.channel_number*8,  kernel_size=4, padding=1, stride=2)# 512x 32x 32
        self.s5_down = down(dropout, in_ch=self.channel_number*8, out_ch=self.channel_number*16, kernel_size=4, padding=1, stride=2)#1024x 16x 16
        self.s4_up   = up  (dropout, in_ch=self.channel_number*16,in_m_ch=self.channel_number*24,out_ch=self.channel_number*4)
        self.s3_up   = up  (dropout, in_ch=self.channel_number*4, in_m_ch=self.channel_number*8, out_ch=self.channel_number*2)
        self.s2_up   = up  (dropout, in_ch=self.channel_number*2, in_m_ch=self.channel_number*4, out_ch=self.channel_number)
        self.s1_up   = up  (dropout, in_ch=self.channel_number,   in_m_ch=self.channel_number*2, out_ch=32)
        #if opt.img_size == 128: self.output  = top_out(dropout, in_ch=32, kernel_size=3 , stride=1, padding=1)
        # if opt.img_size == 64:  
        self.output  = torch.nn.Sequential(Complex_conv2d(32, 1, kernel_size=3, stride=1, padding=1), Complex_batch_norm2d(1), Complex_ReLU(), Complex_dropout2d(dropout), Complex_Sigmoid())
        #if opt.img_size == 64:  self.output  = torch.nn.Sequential(Complex_conv2d(32, 1, kernel_size=3, stride=2, padding=1), Complex_Sigmoid())
        self.Complex_to_Real = Complex_to_Real()

    def forward(self, x_s): #Input 128x128, Output 128x128
        x_s1_skip, x_s = self.s1_down(self.Real_to_Complex(x_s))
        x_s2_skip, x_s = self.s2_down(x_s)
        x_s3_skip, x_s = self.s3_down(x_s)
        x_s4_skip, x_s = self.s4_down(x_s)
        x_s, _ = self.s5_down(x_s) #16x16
        x_s = self.s4_up(x_s, x_s4_skip)
        x_s = self.s3_up(x_s, x_s3_skip)
        x_s = self.s2_up(x_s, x_s2_skip)
        x_s = self.s1_up(x_s, x_s1_skip)
        #x_s_r, x_s_i = self.output(x_s)
        x_s = self.output(x_s)
        return self.Complex_to_Real(x_s) #x_s_r, x_s_i
