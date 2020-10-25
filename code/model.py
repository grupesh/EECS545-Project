import torch.nn as nn
import torch
from torch.nn import init
from collections import OrderedDict
def init_weights(net, init_type='normal', mean = 0.0, init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, mean, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, mean)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, mean)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class FC_net(nn.Module):
    def __init__(self):
        super(FC_net, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(4096, 512)),
            ('dp1', nn.Dropout(p=0.6)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(512, 32)),
            ('dp2', nn.Dropout(p=0.6)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(32, 1)),
            ('dp3', nn.Dropout(p=0.6))
        ]))
        init_weights(self.net, init_type='kaiming', mean=0, init_gain=0.02)

    def forward(self, x):
        x = torch.squeeze(x)
        out = torch.sigmoid(self.net(x))
        return out