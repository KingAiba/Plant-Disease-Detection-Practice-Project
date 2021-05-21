import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import time


def get_device():
    """Change to CUDA if available"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def change_device(data, device):
    """Move tensors to specified device"""
    if isinstance(data, (list, tuple)):
        return [change_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield change_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0).clamp(0, 1))
        plt.show()
        break


data_dir = './Dataset'
# print(os.listdir(data_dir))
# classes = os.listdir(data_dir+"/train")
# print(classes)

# PyTorch datasets
train_ds = ImageFolder(data_dir + '/train', tt.ToTensor())
valid_ds = ImageFolder(data_dir + '/valid', tt.ToTensor())

batch_size = 16
device = get_device()

# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size * 2, num_workers=3, pin_memory=True)

# To GPU
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_channels, out_channels, kernel_size=3, padding=1, stride=1, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(ImageClassificationBase):

    def __init__(self, block, layers, num_classes=32):
        super().__init__()

        # self.inplanes = 64
        #
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)

        self.channel = 64
        # 256 x 256 x 3
        self.conv1 = conv_block(3, 64, pool=True)

        # 128 x 128 x 64
        self.res1 = nn.Sequential(conv_block(64, 64, pool=False), conv_block(64, 64, pool=False))

        # 128 x 128 x 64
        self.conv2 = conv_block(64, 128, pool=True)

        # 64 x 64 x 128
        self.res2 = nn.Sequential(conv_block(128, 128, pool=False), conv_block(128, 128, pool=False))

        # 64 x 64 x 128
        self.conv3 = conv_block(128, 256, pool=True)

        # 32 x 32 x 256
        self.res3 = nn.Sequential(conv_block(256, 256, pool=False), conv_block(256, 256, pool=False))

        # 32 x 32 x 256
        self.conv4 = conv_block(256, 512, pool=True)

        # 16 x 16 x 512
        self.res4 = nn.Sequential(conv_block(512, 512, pool=False),conv_block(512, 512, pool=False))

        # 16 x 16 x 512
        self.classifier = nn.Sequential(nn.MaxPool2d(16),  # 1 x 1 x 512
                                        nn.Flatten(),  # 2048
                                        nn.Dropout(0.2),  # 2048
                                        nn.Linear(512, num_classes))
        # 2048->32

    # def _make_layer(self, block, planes, blocks, stride=1):
    #     downsample = None
    #
    #     if stride != 1 or self.inplanes != planes:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
    #             nn.BatchNorm2d(planes),
    #         )
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #
    #     self.inplanes = planes
    #
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))
    #
    #     return nn.Sequential(*layers)

    def forward(self, xb):
        # x = self.conv1(x)  # 224x224
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)  # 112x112
        #
        # x = self.layer1(x)  # 56x56
        # x = self.layer2(x)  # 28x28
        # x = self.layer3(x)  # 14x14
        # x = self.layer4(x)  # 7x7
        #
        # x = self.avgpool(x)  # 1x1
        # x = torch.flatten(x, 1)  # remove 1 X 1 grid and make vector of tensor shape
        # x = self.fc(x)

        out = self.conv1(xb)

        out = self.res1(out) + out

        out = self.conv2(out)

        out = self.res2(out) + out

        out = self.conv3(out)

        out = self.res3(out) + out

        out = self.conv4(out)

        out = self.res4(out) + out

        out = self.classifier(out)

        return out


# class ResNet50(ImageClassificationBase):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.network = torchvision.models.resnet18()
#         self.network = nn.Sequential(
#
#         )
#         self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
#
#     def forward(self, xb):
#         return self.network(xb)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()


def run():
    # print(len(train_ds.classes))
    # print(device)
    # model = ResNet50(len(train_ds.classes))
    layers = [2, 2, 2, 2]
    model = ResNet(BasicBlock, layers, num_classes=len(train_ds.classes))
    change_device(model, device)

    history = [evaluate(model, valid_dl)]

    epochs = 15
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    TrainTime = time.time()

    history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)

    TrainTime = time.time() - TrainTime
    print("Training Time:" + str(TrainTime / 60))

    plot_accuracies(history)
    plot_losses(history)

    torch.save(model.state_dict(), 'model_plant_disease_resnet_b')

    return


if __name__ == '__main__':
    run()
