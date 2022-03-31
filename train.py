import torch
import argparse
import numpy as np
from torch import nn, optim
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch.nn.functional as F
import random

from torchvision.models import resnet50, vgg16

from skimage import io, transform

from torchvision import transforms, utils
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

import pretrainedmodels

# Local import
from unet import UNet2
from pet_dataset import PetDataset


def get_args():
    parser = argparse.ArgumentParser(description='Run the pipeline')
    parser.add_argument('--epochs', '-e', metavar='E',
                        type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size',
                        metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('--image-size', '-s', dest='ims', type=int, default=128,
                        help='image will be resized to (ims, ims)')
    return parser.parse_args()


def fit(model, dataloader, optimizer, criterion):
    print('-------------Training---------------')
    model.train()
    train_running_loss = 0.0
    counter = 0
    num_batches = int(len(classid) / BATCH_SIZE)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, mask = data["image"].to(DEVICE).float(), data["mask"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(image)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, mask.float())
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss/counter
    return train_loss


def validate(model, dataloader, criterion):
    print("\n--------Validating---------\n")
    model.eval()
    valid_running_loss = 0.0
    counter = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            counter += 1
            image, mask = data["image"].to(
                DEVICE).float(), data["mask"].to(DEVICE)
            outputs = model(image)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, mask.float())
            valid_running_loss += loss.item()
    valid_loss = valid_running_loss/counter
    return valid_loss


if __name__ == '__main__':
    args = get_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    NUM_WORKERS = 4
    IMAGE_HEIGHT = args.ims
    IMAGE_WIDTH = args.ims

    augNtransform = A.Compose([
        A.Resize(128, 128),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        ToTensorV2()
    ])

    pet_dataset = PetDataset(csv_file="/content/gdrive/MyDrive/columbia/cs4995 Deep Learning/project/data/annotations/trainval.txt",
                             root_dir="/content/gdrive/MyDrive/columbia/cs4995 Deep Learning/project/data/images",
                             transform=augNtransform,
                             resizer=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize((128, 128)),
                             ]),
                             # transform = transforms.Compose(
                             #             [transforms.ToTensor(),
                             #              transforms.Resize((128, 128))]),
                             )

    test_dataset = PetDataset(csv_file="/content/gdrive/MyDrive/columbia/cs4995 Deep Learning/project/data/annotations/test.txt",
                              root_dir="/content/gdrive/MyDrive/columbia/cs4995 Deep Learning/project/data/images",
                              transform=transforms.Compose(
                                  [transforms.ToTensor(),
                                   transforms.Resize((128, 128))]),
                              resizer=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Resize((128, 128)),
                              ]))

    classid = pet_dataset.label_data.iloc[:, 1]
    index = np.array(range(len(classid)))
    train_idx, val_idx, y_train, y_test = train_test_split(
        index, classid, test_size=0.1, stratify=classid)

    train_dataset = torch.utils.data.Subset(pet_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(pet_dataset, val_idx)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    train_loss = []
    val_loss = []
    model = UNet2(3, 1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1} of {EPOCHS}")
        train_epoch_loss = fit(model, train_dataloader, optimizer, criterion)
        val_epoch_loss = validate(model, val_dataloader, criterion)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {val_epoch_loss:.4f}')
