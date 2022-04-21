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
import pretrainedmodels

class ResNetV2(nn.Module):
    def __init__(self):
        super(ResNetV2, self).__init__()
        self.model = pretrainedmodels.__dict__["resnet50"](pretrained="imagenet")
        for name, param in self.model.named_parameters():        
            prefix = name[:4]
            if prefix != "last":
              param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2048*4*4, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Linear(in_features=512, out_features=2, bias=True)
        
    def forward(self, x): # 128 x 128
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = self.fc(x.reshape(bs, -1))
        label = self.out(x)
        return label

def train_clf(model, criterion, optimizer, n_epochs=40):
    valid_loss_max = 0.9 # Acc of saved model

    batch_size = 200
    for epoch in range(1, n_epochs):
      k = (epoch-1)*batch_size
      train = [0,1,2,3]
      train_loss = 0.0
      valid_loss = 0.0
      # train the model #
      model.train()
      for batch_idx, sample_batched in enumerate(train_dataloader):
            # importing data and moving to GPU
            image, label = sample_batched['image'].to(DEVICE).float(), sample_batched['species'].to(DEVICE)
            optimizer.zero_grad()
            output=model(image)

            loss = criterion(output, label)
            loss.backward()

            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
                
      # validate the model #
      model.eval()
      for batch_idx, sample_batched in enumerate(val_dataloader):
            image, label = sample_batched['image'].to(DEVICE).float(), sample_batched['species'].to(DEVICE)
            output = model(image)
            _, predicted = torch.max(output, 1)
            train_acc = torch.sum(predicted == label) / len(label)
            
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (train_acc - valid_loss))

      # print training/validation statistics 
      print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Acc: {:.6f}'.format(
          epoch, train_loss, valid_loss))
      
      ## TODO: save the model if validation loss has decreased
      if valid_loss > valid_loss_max:
          torch.save(model.state_dict(), '/content/gdrive/MyDrive/columbia/cs4995 Deep Learning/project/model_clf.pt')
          print('Validation Acc Increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
          valid_loss_max,
          valid_loss))
          valid_loss_max = valid_loss
    # return trained model
    return model