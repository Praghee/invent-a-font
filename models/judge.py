import torch.nn as nn

class JudgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 26)
        )

    def forward(self, x):
        return self.head(self.features(x))