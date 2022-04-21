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
from unet import UNet, UNetCam, UNetCam2, UNetClf
from pet_dataset import PetDataset


def get_args():
    parser = argparse.ArgumentParser(description='Run the pipeline')
    parser.add_argument('--epochs', '-e', metavar='E',
                        type=int, default=20, help='Number of epochs')
    parser.add_argument('--experiment', type=str,
                        default="cam_model", help='experiment name choose form base_model | cam_model | clf_model')                    
    parser.add_argument('--batch-size', '-b', dest='batch_size',
                        metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('--image-size', '-s', dest='ims', type=int, default=128,
                        help='image will be resized to (ims, ims)')
    return parser.parse_args()

class Trainer:
  def __init__(self, experiment_name, model_path=None):
    self.experiment_name = experiment_name

    if experiment_name == "base_model":
      self.model = UNet(3, 1).to(DEVICE)
      self.best_loss = 0.24
    if experiment_name == "cam_model":
      self.model = UNetCam(3, 1).to(DEVICE)
      self.best_loss = 0.24
    if experiment_name == "clf_model":
      self.model = UNetClf(3, 1).to(DEVICE)
      self.best_loss = 0.24

    if model_path is not None:
      self.model.load_state_dict(torch.load(model_path))

    self.optimizer = optim.Adam(self.model.parameters(),lr=1e-5, weight_decay=0.0001)
    self.criterion = nn.BCEWithLogitsLoss()

    self.accuracy = 0.0
    self.iou = 0.0
    self.train_loss = []
    self.val_loss = []
      
  def fit(self, dataloader):
    print('-------------Training---------------')
    self.model.train()
    train_running_loss = 0.0
    counter=0
    num_batches = int(len(classid) // BATCH_SIZE)
    for i, data in tqdm(enumerate(dataloader), total=104):
        counter+=1
        image = data["image"].to(DEVICE)
        mask = data["mask"].to(DEVICE).float()
        cam = data["cam"].to(DEVICE).float()
        self.optimizer.zero_grad()
        if self.experiment_name == "base_model":
          outputs = self.model(image)
        if self.experiment_name == "cam_model":
          outputs = self.model(image, cam)
        if self.experiment_name == "clf_model":
          outputs = self.model(image, image) # -----------------

        outputs =outputs.squeeze(1)
        loss = self.criterion(outputs,mask)
        train_running_loss += loss.item()
        loss.backward()
        self.optimizer.step()
    train_loss = train_running_loss/counter
    return train_loss

  def validate(self, dataloader):
    print("\n--------Validating---------\n")
    self.model.eval()
    valid_running_loss = 0.0
    counter = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            counter+=1
            image = data["image"].to(DEVICE)
            mask = data["mask"].to(DEVICE).float()
            cam = data["cam"].to(DEVICE).float()
            if self.experiment_name == "base_model":
              outputs = self.model(image)
            if self.experiment_name == "cam_model":
              outputs = self.model(image, cam)
            if self.experiment_name == "clf_model":
              outputs = self.model(image, image) # -----------------
            
            outputs =outputs.squeeze(1)
            loss = self.criterion(outputs,mask)
            valid_running_loss += loss.item()

    valid_loss = valid_running_loss/counter
    if valid_loss < self.best_loss:
      print("New Lower Val Loss: {:.2f} -> {:.2f}".format(self.best_loss, valid_loss))
      self.best_loss = valid_loss
      torch.save(self.model.state_dict(), f'/content/gdrive/MyDrive/columbia/cs4995 Deep Learning/project/{self.experiment_name}2.pt')
    return valid_loss

  def train(self, Epochs, train_loader, val_loader):
    for epoch in range(Epochs):
      print(f"Epoch {epoch+1} of {Epochs}")
      train_epoch_loss = self.fit(train_loader)
      val_epoch_loss = self.validate(val_loader)
      self.train_loss.append(train_epoch_loss)
      self.val_loss.append(val_epoch_loss)
      print(f"Train Loss: {train_epoch_loss:.4f}")
      print(f'Val Loss: {val_epoch_loss:.4f}')

  def trainlog(self):
    plt.figure(figsize =(10, 8))
    plt.plot(self.train_loss, label="train loss")
    plt.plot(self.val_loss, label="val loss")

  def test(self, test_loader):
    counter = 0
    self.accuracy = 0.0
    self.iou = 0.0

    with torch.no_grad():
      for i, data in enumerate(test_loader):
          counter+=1
          image = data["image"].to(DEVICE)
          mask = data["mask"].to(DEVICE).float()
          cam = data["cam"].to(DEVICE).float()
          if self.experiment_name == "base_model":
              outputs = self.model(image)
          if self.experiment_name == "cam_model":
            outputs = self.model(image, cam)
          if self.experiment_name == "clf_model":
            outputs = self.model(image, image) # -----------------

          outputs =outputs.squeeze(1)
          outputs[outputs>0.0] = 1.0
          outputs[outputs<=0.0] = 0.0

          correct = torch.sum(outputs == mask)
          self.accuracy += correct / (BATCH_SIZE * 128 * 128)

          tp = torch.sum(outputs * mask)
          self.iou += tp / (outputs.sum() + mask.sum() - tp)

    self.accuracy /= counter
    self.iou /= counter

    print("Acc: ", self.accuracy)
    print("IOU: ", self.iou)

  def visualize(self, test_dataset):
    plt.figure(figsize =(18, 8))
    posi = 1
    for i in random.sample(range(0, 1000), 5):
      data = test_dataset[i]
      image = data['image']
      mask = data['mask']
  
      ax = plt.subplot(3, 5, posi)
      ax.imshow(image.permute(1, 2, 0))
      ax = plt.subplot(3, 5, posi+5)
      ax.imshow(mask)
  
      ax = plt.subplot(3, 5, posi+10)
      img = data["image"].unsqueeze(0).to(DEVICE)
      cam = torch.tensor(data["cam"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
      if self.experiment_name == "base_model":
        output = self.model(img)
      if self.experiment_name == "cam_model":
        output = self.model(img, cam)
      output = torch.squeeze(output)
      output[output>0.0] = 1.0
      output[output<=0.0]=0
      ax.imshow(output.cpu().detach().numpy())
      posi += 1




if __name__ == '__main__':
    args = get_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    NUM_WORKERS = 4
    IMAGE_HEIGHT = args.ims
    IMAGE_WIDTH = args.ims
    EXP_NAME = args.experiment

    augNtransform = A.Compose([
        A.Resize(128,128),
        A.Rotate(limit=35,p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        ToTensorV2()  
    ])

    pet_dataset = PetDataset(csv_file="/content/gdrive/MyDrive/columbia/cs4995 Deep Learning/project/data/annotations/trainval.txt",
                             root_dir="/content/image_compressed",
                             transform = augNtransform
                             )

    test_dataset = PetDataset(csv_file="/content/gdrive/MyDrive/columbia/cs4995 Deep Learning/project/data/annotations/test.txt",
                             root_dir="/content/image_compressed",
                             transform = augNtransform
                            )

    classid = pet_dataset.label_data.iloc[:, 1]
    index = np.array(range(len(classid)))
    train_idx, val_idx, y_train, y_test = train_test_split(index, classid, test_size=0.1, stratify=classid)

    train_dataset = torch.utils.data.Subset(pet_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(pet_dataset, val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


    train_session = Trainer(EXP_NAME)
    train_session.train(Epochs=EPOCHS, train_loader=train_dataloader, val_loader=val_dataloader)
    train_session.trainlog()
    train_session.test(test_dataloader)

