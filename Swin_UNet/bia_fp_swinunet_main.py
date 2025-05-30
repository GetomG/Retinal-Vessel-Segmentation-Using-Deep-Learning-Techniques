# Importing Libraries
import os
import sys
import numpy as np
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from glob import glob
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from swin_unet_model import *
from metrics import *
from data import *

# Set Seeding for Reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Create directories
create_dir("results/files")
create_dir("results/results")
create_dir("results/logs")

# Parameters
H = 512
W = 512
batch_size = 8
lr = 1e-4
epochs = 500

# Load data paths
path = "/home/ubuntu/Ghulam_Murtaza/bia/dataset"
(train_x, train_y), (test_x, test_y) = load_data(path = path, augmented=False)

print(f'length of non-augmented image-mask pairs: {len(train_x), len(train_y)}')
print(f'length of test image-mask pairs: {len(test_x), len(test_y)}')

# Augment data
save_path = os.path.join(path, 'augmented')
augment_data(train_x, train_y, save_path, augment=True)
augment_data(test_x, test_y, save_path, augment=False)

# Load augmented data paths
(train_x, train_y), (test_x, test_y) = load_data(path = path, augmented=True)

print(f'length of augmented image-mask pairs: {len(train_x), len(train_y)}')
print(f'length of test image-mask pairs: {len(test_x), len(test_y)}')

# Split the data
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, random_state=seed)

# Create DataLoaders
train_dataset = SegmentationDataset(train_x, train_y)
valid_dataset = SegmentationDataset(valid_x, valid_y)
test_dataset = SegmentationDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = build_swin_unet(input_shape=(3, H, W))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = model.to(device)

bce_loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training Function
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    iou_score = 0
    dice_score = 0
    f1 = 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = torch.sigmoid(model(x))
        # print(preds.shape)
        loss = criterion(preds, y) #+ dice_loss(y, preds)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        iou_score += iou(y, preds).item()
        dice_score += dice_coef(y, preds).item()
        f1 += f1_score(y, preds).item()
    n = len(loader)
    return epoch_loss / n, iou_score / n, dice_score / n, f1 / n

# Validation Function
def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    iou_score = 0
    dice_score = 0
    f1 = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.sigmoid(model(x))
            loss = criterion(preds, y) #+ dice_loss(y, preds)
            epoch_loss += loss.item()
            iou_score += iou(y, preds).item()
            dice_score += dice_coef(y, preds).item()
            f1 += f1_score(y, preds).item()
    n = len(loader)
    return epoch_loss / n, iou_score / n, dice_score / n, f1 / n

# Initialize wandb
api_key = 'secret'
project_name = 'swin-unet-segmentation'
dataset = 'DRIVE - Digital Retinal Images for Vessel Extraction'
architecture = 'Swin-Unet'

wandb.login(
    key=api_key, relogin=True, force=True
)

wandb.init(
    project=project_name,
    name='Swin-Unet Segmentation',
    
    config={
        "epochs":epochs,
        "batch_size":batch_size,
        "learning_rate":lr,
        "architecture":architecture,
        "loss_function":"BCE+Dice+clDice",
    }
)

# Training Loop
best_val_loss = float("inf")
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss, train_iou, train_dice, train_f1 = train_one_epoch(model, train_loader, optimizer, combined_loss, device)
    val_loss, val_iou, val_dice, val_f1 = evaluate(model, valid_loader, combined_loss, device)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f} | F1: {val_f1:.4f}")
    
    # wandb logging
    wandb.log({
        'Train Loss':train_loss,
        'Val Loss':val_loss,
        'Val IoU':val_iou,
        'Val Dice':val_dice,
        'Val F1-Score':val_f1,
        'Epoch':epoch+1
    })
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "results/files/best_model.pth")

# Save metrics function
import json

def save_metrics(train, val, test, filename="results/logs/metrics.json"):
    metrics = {
        "Train":{
            'Loss':train[0],
            'IoU':train[1],
            'Dice':train[2],
            'F1-Score':train[3],
        },
        
        "Validation":{
            'Loss':val[0],
            'IoU':val[1],
            'Dice':val[2],
            'F1-Score':val[3],
        },
        
        "Test":{
            'Loss':test[0],
            'IoU':test[1],
            'Dice':test[2],
            'F1-Score':test[3],
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

# Load best model and predict
model.load_state_dict(torch.load("results/files/best_model.pth"))
model.eval()

# Evaluate test set metrics
test_loss, test_iou, test_dice, test_f1 = evaluate(model, test_loader, combined_loss, device)
print(f"Test Loss: {test_loss:.4f} | IoU: {test_iou:.4f} | Dice: {test_dice:.4f} | F1: {test_f1:.4f}")

# Save predictions
for i, (x, y) in enumerate(test_loader):
    x = x.to(device)
    with torch.no_grad():
        preds = model(x)
    preds = torch.sigmoid(preds).cpu().numpy() > 0.5
    for j in range(len(preds)):
        pred_mask = (preds[j][0] * 255).astype(np.uint8)
        filename = os.path.basename(test_x[i * batch_size + j])
        cv2.imwrite(f"results/results/{filename}", pred_mask)
        
# Save metrics
save_metrics(
    train=(train_loss, train_iou, train_dice, train_f1),
    val=(val_loss, val_iou, val_dice, val_f1),
    test=(test_loss, test_iou, test_dice, test_f1)
)

wandb.finish()
