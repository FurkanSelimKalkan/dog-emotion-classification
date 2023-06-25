import torch
import torch.nn as nn
from torchvision.models import alexnet
class ImageClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, l2_regularization=0.01):
        super(ImageClassifier, self).__init__()
        self.features = alexnet(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.l2_regularization = l2_regularization

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def l2_loss(self):
        l2_loss = torch.tensor(0.0, device=self.fc[-1].weight.device)
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return l2_loss
