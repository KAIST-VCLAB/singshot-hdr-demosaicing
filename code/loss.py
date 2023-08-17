import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.modules.loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()
        print('Preparing loss function:')

    def forward(self, pred, gt):
        loss_function = nn.L1Loss()
        return loss_function(pred, gt)

    