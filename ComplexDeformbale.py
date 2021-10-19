import torch
import torch.nn
import numpy as np
from ComplexFunctions import *
import torchvision.ops



def param(nnet, Mb=True): return np.round(sum([param.nelement() for param in nnet.parameters()]) / 10**6 if Mb else neles,2)

class ComplexDeformableConv2d(nn.Module):
    ''' [nn.Conv2d] 2D Conv for Complex Numbers '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexDeformableConv2d, self).__init__()
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        kernel_dim = kernel_size[0]*kernel_size[1] 
        self.offset_conv    = RIConv2d(in_channels,kernel_dim*2,   kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.modulator_conv = MagConv2d(in_channels, kernel_dim*1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.regular_conv   = ComplexConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.sigmoid        = nn.Sigmoid()
        nn.init.constant_(self.offset_conv.real_imag_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.mag_conv.weight, 0.0)

    def forward(self, x):
        kernel_offset    = self.offset_conv(x)
        kernal_amplitude = self.sigmoid(self.modulator_conv(x))
        real = (torchvision.ops.deform_conv2d(input=x.real, 
                                          offset=kernel_offset, 
                                          mask=kernal_amplitude,
                                          weight=self.regular_conv.real.weight, 
                                          bias=self.regular_conv.real.bias, 
                                          padding=self.padding,
                                          stride=self.stride) - torchvision.ops.deform_conv2d(input=x.imag, 
                                          offset=kernel_offset, 
                                          mask=kernal_amplitude,
                                          weight=self.regular_conv.imag.weight, 
                                          bias=self.regular_conv.imag.bias, 
                                          padding=self.padding,
                                          stride=self.stride))
        imag = (torchvision.ops.deform_conv2d(input=x.real, 
                                          offset=kernel_offset, 
                                          mask=kernal_amplitude,
                                          weight=self.regular_conv.imag.weight, 
                                          bias=self.regular_conv.imag.bias, 
                                          padding=self.padding,
                                          stride=self.stride) - torchvision.ops.deform_conv2d(input=x.imag, 
                                          offset=kernel_offset, 
                                          mask=kernal_amplitude,
                                          weight=self.regular_conv.real.weight, 
                                          bias=self.regular_conv.real.bias, 
                                          padding=self.padding,
                                          stride=self.stride))
        output = torch.view_as_complex(torch.stack([real, imag],-1))
        return output


class ComplexDeformableConvTranspose2d(nn.Module):
    ''' [nn.ConvTranspose2d] 2D ConvTranpose + 2D Conv for Complex Numbers '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexDeformableConvTranspose2d, self).__init__()
        self.upsample    = ComplexConvTranspose2d(in_channels,  in_channels, kernel_size, stride, padding, dilation, bias=bias, groups=1)
        self.deform_conv = ComplexDeformableConv2d(in_channels, out_channels,(3,3), (1,1), (1,1), (1,1), groups=1, bias=True)
        

    def forward(self, input):
        output = self.deform_conv(self.upsample(input))
        return output
        


if __name__=='__main__':
    cplxmix      = torch.view_as_complex(torch.randn(10,1,300,65,2)).to('cuda')
    cplxmodel1   = ComplexDeformableConv2d(1, 128, (3,3), (1,2), (1,1)).to('cuda')
    cplxmix_out1 = cplxmodel1(cplxmix)
    cplxmodel2   = ComplexDeformableConvTranspose2d(128, 1, (3,3), (1,2), (1,1)).to('cuda')
    cplxmix_out2 = cplxmodel2(cplxmix_out1)
    print('\n\n--------------------------------- Script Inputs and Outputs :: Summary')
    print('Model params (M)  : ', param(cplxmodel1), 'M   ---> Complex Version')
    print('Input Mix audio   : ', cplxmix.real.shape, cplxmix.imag.shape)
    print('Deformable Out    : ', cplxmix_out1.real.shape, cplxmix_out1.imag.shape)
    print('Model params (M)  : ', param(cplxmodel2), 'M')
    print('Tr.Deformable Out : ', cplxmix_out2.real.shape, cplxmix_out2.imag.shape)
    print('--------------------------------------------------------------------------\n')
    print('Done!')