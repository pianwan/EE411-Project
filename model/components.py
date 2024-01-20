from torch import nn
import torch
import torchvision


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class FullConnectLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out


class LocalConnect(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        bs, channels, height, width = x.shape
        x = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = torch.zeros(bs, self.out_channels, out_height, out_width)
        for i in range(out_height):
            for j in range(out_width):
                sh = i * self.stride
                sw = j * self.stride
                eh = sh + self.kernel_size
                ew = sw + self.kernel_size
                local_field = x[:, :, sh:eh, sw:ew]
                out[:, :, i, j] = torch.sum(local_field.unsqueeze(1) * self.weight, dim=(2, 3, 4))
                return out


class LocalConnectLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lc = nn.Sequential(
            LocalConnect(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.lc(x)
        return out
