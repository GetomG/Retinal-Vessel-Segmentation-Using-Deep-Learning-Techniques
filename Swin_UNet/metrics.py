import torch
import torch.nn as nn
import torch.nn.functional as F

smooth = 1e-15
bce_loss = nn.BCELoss()

def iou(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) for each sample in the batch and return the mean.
    
    Args:
        y_true: Ground truth tensor of shape (batch_size, channels, height, width)
        y_pred: Predicted tensor of shape (batch_size, channels, height, width)
    
    Returns:
        Mean IoU over the batch as a scalar tensor
    """
    # Flatten spatial dimensions, preserving batch and channel dimensions
    y_true_f = y_true.view(y_true.shape[0], -1)
    y_pred_f = y_pred.view(y_pred.shape[0], -1)
    
    # Compute intersection and union per sample
    intersection = torch.sum(y_true_f * y_pred_f, dim=1)
    union = torch.sum(y_true_f, dim=1) + torch.sum(y_pred_f, dim=1) - intersection
    
    # Compute IoU per sample and average over the batch
    iou_per_sample = (intersection + smooth) / (union + smooth)
    return torch.mean(iou_per_sample)

def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Dice coefficient for each sample in the batch and return the mean.
    
    Args:
        y_true: Ground truth tensor of shape (batch_size, channels, height, width)
        y_pred: Predicted tensor of shape (batch_size, channels, height, width)
    
    Returns:
        Mean Dice coefficient over the batch as a scalar tensor
    """
    # Flatten spatial dimensions, preserving batch and channel dimensions
    y_true_f = y_true.view(y_true.shape[0], -1)
    y_pred_f = y_pred.view(y_pred.shape[0], -1)
    
    # Compute intersection per sample
    intersection = torch.sum(y_true_f * y_pred_f, dim=1)
    
    # Compute Dice coefficient per sample and average over the batch
    dice_per_sample = (2. * intersection + smooth) / (torch.sum(y_true_f, dim=1) + torch.sum(y_pred_f, dim=1) + smooth)
    return torch.mean(dice_per_sample)

def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Dice loss as 1 minus the mean Dice coefficient.
    
    Args:
        y_true: Ground truth tensor of shape (batch_size, channels, height, width)
        y_pred: Predicted tensor of shape (batch_size, channels, height, width)
    
    Returns:
        Scalar loss value (average Dice loss over the batch)
    """
    return 1.0 - dice_coef(y_true, y_pred)

def soft_skel(x, iter_=3):
    def erod(x):
        return 1.0 - nn.functional.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
    skel = x
    for _ in range(iter_):
        skel = skel * erod(skel)
    return skel

def soft_cldice_loss(y_true, y_pred, iter_=3, smooth=1.0):
    y_pred = torch.clamp(y_pred, 0.0, 1.0)
    S_true = soft_skel(y_true, iter_)
    S_pred = soft_skel(y_pred, iter_)
    
    inter1 = torch.sum(y_pred * S_true)
    inter2 = torch.sum(y_true * S_pred)
    sum1 = torch.sum(y_pred) + torch.sum(S_true)
    sum2 = torch.sum(y_true) + torch.sum(S_pred)
    
    C = (2.0 * inter1 + smooth) / (sum1 + smooth)
    D = (2.0 * inter2 + smooth) / (sum2 + smooth)
    return 1.0 - (C * D)

def combined_loss(y_true, y_pred):
    bce = bce_loss(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred)
    cld = soft_cldice_loss(y_true, y_pred)
    return bce + 0.5*dsc + 0.5*cld

def f1_score(y_true, y_pred):
    y_pred = y_pred > 0.5
    y_true = y_true > 0.5
    tp = (y_pred & y_true).sum().float()
    fp = (y_pred & ~y_true).sum().float()
    fn = (~y_pred & y_true).sum().float()
    return (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)

