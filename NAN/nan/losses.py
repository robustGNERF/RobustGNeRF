from functools import partial

import torch

from nan.utils.general_utils import TINY_NUMBER

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGG(nn.Module):
    """VGG/Perceptual Loss
    
    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output

    """
    def __init__(self, conv_index: str = '54'):

        super(VGG, self).__init__()
        vgg_features = torchvision.models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)
        #self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False

def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute VGG/Perceptual loss between Super-Resolved and High-Resolution

        Parameters
        ----------
        sr : torch.Tensor
            Super-Resolved model output tensor
        hr : torch.Tensor
            High-Resolution image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr

        """
        def _forward(x):
            #x = self.sub_mean(x)
            x = self.vgg(x)
            return x
        vgg_sr = _forward(sr)

        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss



def mean_with_mask(x, mask):
    return torch.sum(x * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


def loss_with_mask(fn, x, y, mask=None):
    """
    @param x: img 1, [(...), 3]
    @param y: img 2, [(...), 3]
    @param mask: optional, [(...)]
    @param fn: loss function
    """
    loss = fn(x, y)
    if mask is None:
        return torch.mean(loss)
    else:
        return mean_with_mask(loss, mask)


l2_loss  = partial(loss_with_mask, lambda x, y: (x - y) ** 2)
l1_loss  = partial(loss_with_mask, lambda x, y: torch.abs(x - y))
gen_loss = partial(loss_with_mask, lambda x, y: x, 0)