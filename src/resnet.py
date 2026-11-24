import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet


class ResNet18(ResNet):
    def __init__(self, in_channels=39, embedding_size=2):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=1)
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.fc_reg = nn.Linear(512, embedding_size)
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, images):
        features = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        embedding = self.fc_reg(torch.flatten(self.avgpool(features), 1))
        return self.fc(embedding).flatten(), embedding
