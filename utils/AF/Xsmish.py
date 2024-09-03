"""
Script based on:
Wang, Xueliang, Honge Ren, and Achuan Wang.
 "Smish: A Novel Activation Function for Deep Learning Methods.
 " Electronics 11.4 (2022): 540.
smish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + sigmoid(x)))
"""

# import pytorch
import torch
import torch.nn.functional as F
from torch import nn

import sys
sys.path.append('/home/kongxz/workspace/Crepe/model/utils')

# import activation functions
# import utils.AF.Fsmish as Func
#from .Fsmish import smish

@torch.jit.script
def smish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(sigmoid(x))))
    See additional documentation for mish class.
    """
    return input * torch.tanh(torch.log(1+torch.sigmoid(input)))


class Smish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Mish.html
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return smish(input)