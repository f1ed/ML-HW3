#################
# Data:2020-05-14
# Author: Fred Lau
# ML-Lee: HW3 :  Convolutional Neural Network
###########################################################
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time


# Read image
def readfile(path, label):
    # label is a boolean variable: whether to return y
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split('_')[0])
    if label:
        return x, y
    else:
        return x


# read training/ validation/ testing set
workspace_dir = './food-11'
with open('cnn_init.txt', 'w') as f:
    f.writelines("Reading data\n")
    train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
    f.write("Size of training data = {}\n".format(len(train_x)))
    val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
    f.write("Size of validation data = {}\n".format(len(val_x)))
    test_x = readfile(os.path.join(workspace_dir, "testing"), False)
    f.write("Size of testing data = {}".format(len(test_x)))

# training: data argumentation 数据增强
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(),
    transforms.ToPILImage()
])

# testing: no data argumentation, ndarray to tensor
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor
])


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        Dataset.__init__(self, x, y)  # inherit constructor
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.Y[index]
            return X, Y
        else:
            return X


batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transforms)
val_set = ImgDataset(val_x, val_y, test_transforms)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# Model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # input : C H W = [3, 128, 128]
        self.cnn = nn.Sequential(
            # Conv2d(in_channels, out_channels, kernel_size, stride, zero-padding)
            # (H_in + 2*padding - kernel)/stride + 1
            nn.Conv2d(3, 64, 3, 1, 1),  # output: C H W = [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # MaxPool2d(kernel_size, stride, padding)
            nn.MaxPool2d(2, 2, 0),  # output: C H W = [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # C H W = [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # C H W = [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # C H W = [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # C H W = [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # C H W = [512, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # C H W = [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # C H W = [512, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)  # C H W = [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        # flatten
        out = out.view(out.size()[0], -1)
        return self.fc(out)

# training
model = Classifier().cuda()
loss = nn.CrossEntropyLoss()  # Classification task
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 30
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # assure dropout is open
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().))
