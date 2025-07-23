import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# 1. Basic autograd example 1 
# Create tensors.
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)