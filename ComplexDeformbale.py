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
        


if __name__=='__main__':
    cplxmix     = torch.view_as_complex(torch.randn(10,64,300,257,2)).to('cuda')
    cplxmodel   = ComplexDeformableConv2d(64, 256, (3,3), 1, 1).to('cuda')
    cplxmix_out = cplxmodel(cplxmix)
    print('\n\n--------------------------------- Script Inputs and Outputs :: Summary')
    print('Model params (M) : ', param(cplxmodel), 'M   ---> Complex Version')
    print('Input Mix audio  : ', cplxmix.real.shape, cplxmix.imag.shape)
    print('Deformable Out   : ', cplxmix_out.real.shape, cplxmix_out.imag.shape)
    print('--------------------------------------------------------------------------\n')
    print('Done!')