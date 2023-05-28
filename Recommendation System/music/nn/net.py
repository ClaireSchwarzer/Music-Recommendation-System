import torch.nn as nn

# AlexNet network
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Convolutional layer: Input channels = 1, Output channels = 64, Kernel size = 11x11, Stride = 4, Padding = 2
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            # ReLU activation function
            nn.ReLU(inplace=True),
            # Max pooling
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Flatten
        self.flatten = nn.Flatten()
        # Classifier
        self.classifier = nn.Sequential(
            # Linear classifier (fully connected layer)
            nn.Linear(12288, 1024),
            nn.ReLU(inplace=True),
            # Dropout
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(1024, num_classes),
        )
    
    # Forward pass
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x