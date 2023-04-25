import torch
import torch.nn.functional as F
from dice import dice_loss
class CombinedLoss(torch.nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, output, target):
        dice = dice_loss(target,output,output.shape[-1]) # 计算Dice损失
        ce_loss = F.cross_entropy(output, target) # 计算交叉熵损失
        loss = self.weight_dice * dice + self.weight_ce * ce_loss
        return loss