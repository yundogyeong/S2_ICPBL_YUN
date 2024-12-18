import torch
import torch.nn as nn

class BrightnessClassifier(nn.Module):
    def __init__(self):
        super(BrightnessClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 30 * 30, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 출력 뉴런: 3개 (어두움, 중간, 밝음)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
