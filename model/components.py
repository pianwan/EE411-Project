import torch
from torch import nn
from torch.nn.modules.utils import _pair


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


# class LocalConnect(nn.Module):
#     def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
#         super(LocalConnect, self).__init__()
#         output_size = _pair(output_size)
#         self.weight = nn.Parameter(
#             torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
#         )
#         if bias:
#             self.bias = nn.Parameter(
#                 torch.randn(1, out_channels, output_size[0], output_size[1])
#             )
#         else:
#             self.register_parameter('bias', None)
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#
#     def forward(self, x):
#         _, c, h, w = x.size()
#         kh, kw = self.kernel_size
#         dh, dw = self.stride
#         x = x.unfold(2, kh, dh).unfold(3, kw, dw)
#         x = x.contiguous().view(*x.size()[:-2], -1)
#         # Sum in in_channel and kernel_size dims
#         out = (x.unsqueeze(1) * self.weight).sum([2, -1])
#         if self.bias is not None:
#             out += self.bias
#         return out

# class LocalConnect(nn.Module):
#     def _compute_output_img_size(self, input_img_size):
#         output_img_width = input_img_size[0] + 2 * self.padding - self.kernel_size
#         output_img_width = int(output_img_width / self.stride) + 1
#         output_img_height = input_img_size[1] + 2 * self.padding - self.kernel_size
#         output_img_height = int(output_img_height / self.stride) + 1
#         return (output_img_width, output_img_height)
#
#     def __init__(self, in_channels, out_channels, input_size, kernel_size, stride, padding):
#         super(LocalConnect, self).__init__()
#         self.input_size = _pair(input_size)
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.weight = nn.Parameter(
#             torch.randn(1, out_channels, in_channels, input_size[0], input_size[1], kernel_size ** 2)
#         )
#
#     def forward(self, x):
#         _, c, h, w = x.size()
#         kh, kw = self.kernel_size
#         dh, dw = self.stride
#         x = x.unfold(2, kh, dh).unfold(3, kw, dw)
#         x = x.contiguous().view(*x.size()[:-2], -1)
#         # Sum in in_channel and kernel_size dims
#         out = (x.unsqueeze(1) * self.weight).sum([2, -1])
#         if self.bias is not None:
#             out += self.bias
#         return out

class LocalConnect(nn.Module):
    def _compute_output_img_size(self, input_img_size):

        output_img_w = input_img_size[0] + 2 * self.padding[0] - self.kernel_size[0]
        output_img_w = int(output_img_w / self.stride[0]) + 1
        output_img_h = input_img_size[1] + 2 * self.padding[1] - self.kernel_size[1]
        output_img_h = int(output_img_h / self.stride[1]) + 1

        return output_img_w, output_img_h

    def __init__(self, in_channels, out_channels, input_img_size, kernel_size, stride, padding):
        super(LocalConnect, self).__init__()
        self.input_img_size = (input_img_size, input_img_size)
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        output_img_size = self._compute_output_img_size(self.input_img_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_img_size[0], output_img_size[1], kernel_size ** 2)
        )

    def forward(self, x):
        _, c, h, w = x.size()
        kernel_w, kernel_h = self.kernel_size
        stride_w, stride_h = self.stride
        padding_two = self.padding + self.padding
        x = nn.functional.pad(x, padding_two)
        x = x.unfold(2, kernel_h, stride_h).unfold(3, kernel_w, stride_w)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])

        return out

class LocalConnectLayer(nn.Module):
    def __init__(self, in_channels, out_channels, input_img_size, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_img_size = input_img_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lc = nn.Sequential(
            LocalConnect(
                in_channels=in_channels,
                out_channels=out_channels,
                input_img_size=input_img_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _compute_output_img_size(self, input_img_size):
        return self.lc[0]._compute_output_img_size(input_img_size)

    def forward(self, x):
        out = self.lc(x)
        return out

#
# class LocalConnect(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
#
#     def forward(self, x):
#         bs, channels, height, width = x.shape
#         x = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
#         out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
#         out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
#         out = torch.zeros(bs, self.out_channels, out_height, out_width)
#         for i in range(out_height):
#             for j in range(out_width):
#                 sh = i * self.stride
#                 sw = j * self.stride
#                 eh = sh + self.kernel_size
#                 ew = sw + self.kernel_size
#                 local_field = x[:, :, sh:eh, sw:ew]
#                 out[:, :, i, j] = torch.sum(local_field.unsqueeze(1) * self.weight, dim=(2, 3, 4))
#                 return out
#
#
# class LocalConnectLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.lc = nn.Sequential(
#             LocalConnect(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         out = self.lc(x)
#         return out
