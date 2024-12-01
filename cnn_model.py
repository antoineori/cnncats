import torch
import torch.nn as nn


class CatCNN(nn.Module):
    def __init__(self):
        super(CatCNN, self).__init__()

        # Initial convolution block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block 1
        self.conv2a = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(32)
        self.prelu2a = nn.PReLU()
        self.conv2b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn2b = nn.BatchNorm2d(32)
        self.prelu2b = nn.PReLU()

        self.residual1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=2)

        # Residual block 2
        self.conv3a = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn3a = nn.BatchNorm2d(32)
        self.prelu3a = nn.PReLU()
        self.conv3b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(32)
        self.prelu3b = nn.PReLU()

        self.residual2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1)

        # Average pooling and fully connected layer
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(512, 7)

    def forward(self, x):
        # Initial block
        x = self.pool1(self.prelu1(self.bn1(self.conv1(x))))

        # Residual block 1
        residual = self.residual1(x)
        x = self.prelu2a(self.bn2a(self.conv2a(x)))
        x = self.prelu2b(self.bn2b(self.conv2b(x)))
        x += residual

        # Residual block 2
        residual = self.residual2(x)
        x = self.prelu3a(self.bn3a(self.conv3a(x)))
        x = self.prelu3b(self.bn3b(self.conv3b(x)))
        x += residual

        # Global average pooling and fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Test the model architecture
if __name__ == "__main__":
    model = CatCNN()
    print(model)

    # Verify parameter count
    from torchinfo import summary
    print(summary(model, input_size=(1, 3, 224, 224)))
