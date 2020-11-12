import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # input size is (batch, 1, 256, 256)
        # target size is (batch, 1, 256, 256)
        
        #https://pytorch.org/docs/master/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        bce = F.binary_cross_entropy_with_logits(input, target)
        
        smooth = 1e-5
        
        input = torch.sigmoid(input)
        #input still same shape
        
        #num will be equal to batch_size (first element of tensor)
        num = target.size(0)
        
        #basically flattens the array of predictions
        # ie. goes from (4, 1, 256, 256) to (4, 65536)
        input = input.view(num, -1)
        
        target = target.view(num, -1)
        
        intersection = (input * target)
        
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        
        dice = 1 - dice.sum() / num
        
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
