import torch
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class MnistResNet(ResNet):
    def __init__(self, block, layers, num_classes=10):
        super(MnistResNet, self).__init__(block=block, layers=layers, num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fd = 512

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = x  # Activations from the last pooling layer
        logits = self.fc(x)  # Final classification layer output
        return features, logits




def ResNet18(num_classes=10):
    return MnistResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return MnistResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return MnistResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return MnistResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return MnistResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
