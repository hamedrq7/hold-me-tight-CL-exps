import torch 
import torch.nn as nn 
from collections import OrderedDict

class MNIST_TRADES(nn.Module):
    def __init__(self, zero_bias=False, drop=0.5):
        super(MNIST_TRADES, self).__init__()

        self.num_channels = 1
        self.num_labels = 10
        self.feat_dim = 200

        self.conv1 = nn.Conv2d(self.num_channels, 32, 3)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.relu2 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.relu3 = nn.ReLU(True)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.relu4 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 200)
        self.relu5 = nn.ReLU(True)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(200, 200)
        self.relu6 = nn.ReLU(True)
        self.fc3 = nn.Linear(200, self.num_labels, bias=False if zero_bias else True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.xavier_normal_(self.fc3.weight)

        self.grad_layer = [self.maxpool1, self.maxpool2]
        self.fd = 200

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.maxpool1(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.maxpool2(x)

        x = x.view(-1, 64 * 4 * 4)
        feats = self.relu5(self.fc1(x))
        feats = self.drop(feats)
        feats = self.relu6(self.fc2(feats))
        logits = self.fc3(feats)

        return feats, logits
