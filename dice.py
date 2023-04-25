import torch
import torch.nn.functional as F
def dice_coef(y_true,y_pred,num_classes =26,smooth=1e-7):
    y_pred = torch.reshape(y_pred,[-1,num_classes])
    y_pred = y_pred.to(torch.int64)
    y_true = torch.reshape(y_true,[-1])
    y_true = y_true.to(torch.int64)
    y_true = F.one_hot(y_true,num_classes)
    y_true = y_true[:,1:]
    y_pred = y_pred[:,1:]
    inter = torch.sum(y_true*y_pred,dim=0)
    X = torch.sum(y_pred,dim=0)
    Y = torch.sum(y_true,dim=0)
    dice = (2.0*inter)/(X+Y+smooth)
    return dice
def dice_loss(y_true,y_pred,num_classes=26,smooth=1e-7):
    dice = dice_coef(y_true,y_pred,num_classes,smooth)
    return 1.0-torch.mean(dice)

# y_pred = torch.randint(2,(20,256,256,26))
# y_true = torch.randint(2,(20,256,256,1))
# print(y_pred.shape[-1])
# # print(dice_loss(y_true,y_pred))

