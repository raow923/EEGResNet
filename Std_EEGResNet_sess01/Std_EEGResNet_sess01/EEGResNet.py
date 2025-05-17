"""
一句话：构建网络：
    # 输入形状：bs,4,9,512

NiceParas:
    # 1 lr = 8,9,10e-4
    # 2 elf.dropout = nn.Dropout(0.25)
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 3), (1, stride), (0, 1)),
            nn.BatchNorm2d(outchannel),
            nn.Conv2d(outchannel, outchannel, (1, 3), (1, 1), (0, 1)),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, (1, 1), (1, stride)),
                nn.BatchNorm2d(outchannel)
            )
        self.elu = nn.ELU(alpha=1)

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = self.elu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=4):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(  # 输入bs,4,9,512
            nn.Conv2d(4, 16, (9, 3), (1, 1), (0, 1)),  # 形状bs,16,1,512
            nn.BatchNorm2d(16),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 1, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 16, 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 32, 1, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 32, 1, stride=2)
        self.layer5 = self.make_layer(ResidualBlock, 64, 1, stride=2)
        self.layer6 = self.make_layer(ResidualBlock, 64, 1, stride=2)
        self.fc = nn.Linear(64 * 8, num_classes)
        self.dropout = nn.Dropout(0.25)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.layer1(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.dropout(out)
        out = self.layer5(out)
        out = self.dropout(out)
        out = self.layer6(out)
        out = self.dropout(out)

        out = torch.flatten(out, 1, 3)
        out = self.fc(out)

        return out
