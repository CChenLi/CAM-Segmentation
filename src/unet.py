""" UNet Based on https://github.com/milesial/Pytorch-UNet#training """

import torch
import torch.nn as nn
import torch.nn.functional as F
from Rescnn import ResNetV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Att(nn.Module):
  """Downscaling CAM and calculate attention score"""
  def __init__(self, in_channels, out_channels):
    super(Att, self).__init__()
    self.filters = out_channels
    self.model = nn.Sequential(
        nn.Linear(in_channels, 2048),
        nn.ReLU(),
        nn.Linear(2048, out_channels),
        nn.Softmax(dim=1)
    )
  def forward(self, x):
    bs = x.shape[0]
    out = self.model(x.view(bs, -1)) * self.filters
    return out

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, att=None):
        bs = x1.shape[0]
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1) # new Channel number = x1 channel number + x2.channel number
        if att is not None:
            x = x.view(bs, 128, -1)
            x = (att * x).view(bs, 128, 128, 128)
            x = F.relu(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        print("Initialize UNet")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.inc_cam = DoubleConv(1, 16)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x1 = self.inc(x) # [BS, 64, 128, 128]
        x2 = self.down1(x1) # [BS, 128, 64, 64]
        x3 = self.down2(x2) # [BS, 256, 32, 32]
        x4 = self.down3(x3) # [BS, 512, 16, 16]
        x5 = self.down4(x4) # [BS, 1024, 8, 8]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetCam(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetCam, self).__init__()
        print("Initialize UNetCam")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.inc_cam = DoubleConv(1, 16)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.down_cam1 = Down(16, 32)
        self.down_cam2 = Down(32, 64)
        self.down_cam3 = Down(64, 128)
        self.down_cam4 = Down(128, 64)
        self.attention = Att(4096, 128)

    def forward(self, x, cam):
        bs, _, _, _ = x.shape
        x1 = self.inc(x) # [BS, 64, 128, 128]
        x2 = self.down1(x1) # [BS, 128, 64, 64]
        x3 = self.down2(x2) # [BS, 256, 32, 32]
        x4 = self.down3(x3) # [BS, 512, 16, 16]
        x5 = self.down4(x4) # [BS, 1024, 8, 8]
        
        cam1 = self.inc_cam(cam)
        cam2 = self.down_cam1(cam1)
        cam3 = self.down_cam2(cam2)
        cam4 = self.down_cam3(cam3)
        cam5 = self.down_cam4(cam4)
        attention = self.attention(cam5).view(1, bs, 128, 1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2).view(bs, 128, -1)
        # Apply attention to filters right before concat with original input
        x = (attention * x).view(bs, 128, 64, 64)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetCam2(nn.Module):
    # Utilize CAM for feature selection
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetCam2, self).__init__()
        print("Initialize UNetCam2")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.inc_cam = DoubleConv(3, 16)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.down_cam1 = Down(16, 32)
        self.down_cam2 = Down(32, 64)
        self.down_cam3 = Down(64, 128)
        self.down_cam4 = Down(128, 128)
        self.attention = Att(4096 * 2, 128)

    def forward(self, x, cam):
        bs, _, _, _ = x.shape
        x1 = self.inc(x) # [BS, 64, 128, 128]
        x2 = self.down1(x1) # [BS, 128, 64, 64]
        x3 = self.down2(x2) # [BS, 256, 32, 32]
        x4 = self.down3(x3) # [BS, 512, 16, 16]
        x5 = self.down4(x4) # [BS, 1024, 8, 8]
        
        cam1 = self.inc_cam(cam)
        cam2 = self.down_cam1(cam1)
        cam3 = self.down_cam2(cam2)
        cam4 = self.down_cam3(cam3)
        cam5 = self.down_cam4(cam4)
        attention = self.attention(cam5).view(1, bs, 128, 1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1, attention)
        logits = self.outc(x)
        return logits

class UNetClf(nn.Module):
    # Utilize pretrained classifier for feature selection
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetClf, self).__init__()
        print("Initialize UNetClf")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.classifier = ResNetV2().to(DEVICE)
        self.classifier.load_state_dict(torch.load('/content/gdrive/MyDrive/columbia/cs4995 Deep Learning/project/model_clf.pt'))
        for param in self.classifier.parameters():
          param.requires_grad = False

        self.attention = Att(512, 128)

    def forward(self, x, cam):
        bs, _, _, _ = x.shape
        x1 = self.inc(x) # [BS, 64, 128, 128]
        x2 = self.down1(x1) # [BS, 128, 64, 64]
        x3 = self.down2(x2) # [BS, 256, 32, 32]
        x4 = self.down3(x3) # [BS, 512, 16, 16]
        x5 = self.down4(x4) # [BS, 1024, 8, 8]
        
        cam = self.classifier.model.features(cam)
        cam = self.classifier.fc(cam.reshape(bs, -1))
        attention = self.attention(cam).view(1, bs, 128, 1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1, attention)
        logits = self.outc(x)
        return logits