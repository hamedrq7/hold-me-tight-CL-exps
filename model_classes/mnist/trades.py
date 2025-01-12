import torch 
import torch.nn as nn 
from collections import OrderedDict

class MNIST_TRADES(nn.Module):
    def __init__(self, zero_bias=False, drop=0.5):
        super(MNIST_TRADES, self).__init__()

        self.num_channels = 1
        self.num_labels = 10
        self.feat_dim = 200

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            # ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        self.fc3 = nn.Linear(200, self.num_labels, bias=False if zero_bias else True)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.xavier_normal_(self.fc3.weight)

        self.grad_layer = []
        self.fd = 200 

    def normalize_weights(self, ): 
        with torch.no_grad():
            self.fc3.weight.data = 1. * torch.nn.functional.normalize(self.fc3.weight.data, dim=1)

        # norms = torch.norm(self.fc3.weight.data, p=2, dim=1)
        # self.fc3.weight.data = 1. * (self.fc3.weight.data.T / norms).T

    def get_softmax_weights(self,):
        return self.fc3.weight.data 
    
    def get_penultimate_layer(self, ):
        return self.fc3

    def forward(self, x):
        x = self.feature_extractor(x)
        feats = self.classifier(x.view(-1, 64 * 4 * 4))
        # self.normalize_weights()
        # feats = torch.nn.functional.normalize(feats, dim=1)
        # logits = 15.0 * self.fc3(feats)
        logits = self.fc3(feats)
        return feats, logits
