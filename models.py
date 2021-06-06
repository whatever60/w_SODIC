import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, ipt):
        return self.conv(ipt)


class Unet(nn.Module):
    input_size = (5, 5)
    target_size = (15, 14)
    def __init__(self, in_ch, out_ch, num_heights):
        super().__init__()
        height_dim = 10
        height_channels = 10
        self.conv1 = DoubleConv(in_ch + height_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up5 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv6 = DoubleConv(256, 128)
        self.up7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(64, 64, 4, stride=4)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, (6, 7))
        self.num_heights = num_heights
        self.height_embedding = nn.Embedding(num_heights, height_channels)
        self.height_conv = nn.ConvTranspose2d(height_dim, height_channels, kernel_size=self.input_size, stride=1, padding=0)

    def forward(self, x, height):
        height_emb = self.height_embedding(height).view(x.shape[0], -1, 1, 1)
        c1 = self.conv1(torch.cat([x, self.height_conv(height_emb)], dim=1))
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        up_4 = self.up4(c3)
        up_5 = self.up5(c2)
        merge5 = torch.cat([up_5, up_4], dim=1)
        c6 = self.conv6(merge5)
        up_7 = self.up7(c6)
        up_8 = self.up8(c1)
        merge8 = torch.cat([up_8, up_7], dim=1)
        c9 = self.conv9(merge8)
        c10 = self.conv10(c9)

        return c10


def test_unet():
    model = Unet(5, 5, 12)
    height = torch.randint(10, size=(10,))
    input_ = torch.randn(10, 5, 5, 5)
    rprint(model(input_, height).shape)


if __name__ == '__main__':
    from rich import print as rprint
    from rich.traceback import install
    install()
    
    test_unet()
    