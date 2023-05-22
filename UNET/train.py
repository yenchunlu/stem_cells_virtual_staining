
import os
import time
from operator import add
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score


from data import DriveDataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
import numpy as np

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().detach().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        score = calculate_metrics(y, y_pred)
        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
        metrics_score = list(map(add, metrics_score, score))

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss, metrics_score

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    save_dir = f"output/epoch_{epoch}"
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_X = sorted(glob("/Users/lu992/Documents/Unet implementation/new_data/train/images/*")) # *: get all the images paths
    train_y = sorted(glob("/Users/lu992/Documents/Unet implementation/new_data/train/masks/*"))

    valid_X = sorted(glob("/Users/lu992/Documents/Unet implementation/new_data/test/images/*"))
    valid_y = sorted(glob("/Users/lu992/Documents/Unet implementation/new_data/test/masks/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_X)} - Valid: {len(valid_X)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_X, train_y)
    valid_dataset = DriveDataset(valid_X, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   ## use GPU if available, else use CPU
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, metric_score = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        jaccard = metric_score[0]/len(train_X)
        f1 = metric_score[1]/len(train_X)
        recall = metric_score[2]/len(train_X)
        precision = metric_score[3]/len(train_X)
        acc = metric_score[4]/len(train_X)


        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\tTrain:\n\tJaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)