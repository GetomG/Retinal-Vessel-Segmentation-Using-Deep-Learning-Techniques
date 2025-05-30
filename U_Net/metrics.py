import tensorflow as tf
from tensorflow.keras import backend as K

smooth = 1e-15

def iou(y_true, y_pred):
    """
    Calculate the Intersection over Union (IoU) for each sample in the batch and return the mean.
    
    Args:
        y_true: Ground truth tensor of shape (batch_size, height, width, channels)
        y_pred: Predicted tensor of shape (batch_size, height, width, channels)
    
    Returns:
        Mean IoU over the batch as a scalar tensor
    """
    # Flatten spatial dimensions, preserving batch dimension
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    
    # Compute intersection and union per sample
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    union = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) - intersection
    
    # Compute IoU per sample and average over the batch
    iou_per_sample = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou_per_sample)

def dice_coef(y_true, y_pred):
    """
    Calculate the Dice coefficient for each sample in the batch and return the mean.
    
    Args:
        y_true: Ground truth tensor of shape (batch_size, height, width, channels)
        y_pred: Predicted tensor of shape (batch_size, height, width, channels)
    
    Returns:
        Mean Dice coefficient over the batch as a scalar tensor
    """
    # Flatten spatial dimensions, preserving batch dimension
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    
    # Compute intersection per sample
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    
    # Compute Dice coefficient per sample and average over the batch
    dice_per_sample = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth)
    return tf.reduce_mean(dice_per_sample)

def dice_loss(y_true, y_pred):
    """
    Calculate the Dice loss as 1 minus the mean Dice coefficient.
    
    Args:
        y_true: Ground truth tensor of shape (batch_size, height, width, channels)
        y_pred: Predicted tensor of shape (batch_size, height, width, channels)
    
    Returns:
        Scalar loss value (average Dice loss over the batch)
    """
    return 1.0 - dice_coef(y_true, y_pred)