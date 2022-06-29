import torch
from torch import nn

# "M": Max-Pool2d
# Tuple: (kernel_size, num_filters, stride, padding)
# List: list of tuples where last int indicates number of times to repeat these tuples sequentially
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.leakyrelu(out)
        return out


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        out = self.darknet(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fcs(out)
        return out

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels  # = 3

        for x in architecture:
            if type(x) == tuple:
                layers.append(CNNBlock(
                    in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]))

                in_channels = x[1]
            elif type(x) == str:
                layers.append(torch.nn.MaxPool2d(
                    kernel_size=(2, 2), stride=2))
            elif type(x) == list:
                conv1 = x[0]  # tuple
                conv2 = x[1]  # tuple
                num_repeats = x[2]  # int

                for _ in range(num_repeats):
                    layers.append(CNNBlock(
                        in_channels, out_channels=conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]))

                    layers.append(CNNBlock(
                        conv1[1], out_channels=conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]))

                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        # in the Original paper this 512 should be 4096 but due to resources I'll reduce it
        hidden_dim = 512

        return torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(1024 * S * S, hidden_dim),
            torch.nn.Dropout(0.0),
            torch.nn.LeakyReLU(0.1),
            # Note that (* 5) stands for (p_c, x, y, h, w)
            torch.nn.Linear(hidden_dim, S * S * (C + (B * 5))
                            ),  # --> [S, S, 30]
        )


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


if __name__ == "__main__":
    def test(S=7, B=2, C=20):
        model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
        x = torch.randn((2, 3, 448, 448))
        print(model(x).shape)  # --> [2, 1470=(7*7*30)]

    test()
