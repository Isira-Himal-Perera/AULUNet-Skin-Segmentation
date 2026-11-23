import torch
import torch.nn as nn
import torch.nn.functional as F


class PointConv(nn.Module):
    """PointConv block: 1x1 Conv -> BatchNorm2d -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super(PointConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseConvBlock(nn.Module):
    """Lightweight depthwise 3x3 conv block: DWConv3x3 -> BatchNorm2d -> ReLU"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AULUNetBaseline(nn.Module):
    """Minimal U-Net baseline with channel widths [5, 10, 15, 25, 35, 60],
    using depthwise 3x3 convs instead of AKF, no RGF, and direct skip fusion instead of LSG.
    """

    def __init__(
        self, channels=[5, 10, 15, 25, 35, 60], in_channels=3, out_channels=1
    ):
        super().__init__()
        self.enc_channels = channels

        # Stage 0: project RGB to channels[0], then apply depthwise 3x3 conv
        self.init_conv = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, padding=1, bias=False
        )
        self.init_dw = DepthwiseConvBlock(channels[0])

        # Encoder stages: AvgPool2d -> DepthwiseConvBlock -> PointConv (channel change)
        self.enc_dws = nn.ModuleList(
            [
                DepthwiseConvBlock(channels[i])
                for i in range(len(channels) - 1)
            ]
        )
        self.enc_pointconvs = nn.ModuleList(
            [
                PointConv(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )

        # Bottleneck: depthwise 3x3 conv (replaces RGF)
        self.bottleneck_dw = DepthwiseConvBlock(channels[-1])

        # Decoder stages: PointConv -> concat skip -> PointConv -> DepthwiseConvBlock
        self.dec_pointconvs = nn.ModuleList(
            [
                PointConv(channels[i + 1], channels[i])
                for i in range(len(channels) - 2, -1, -1)
            ]
        )
        self.dec_fuse_pointconvs = nn.ModuleList(
            [
                PointConv(channels[i] * 2, channels[i])
                for i in range(len(channels) - 2, -1, -1)
            ]
        )
        self.dec_dws = nn.ModuleList(
            [
                DepthwiseConvBlock(channels[i])
                for i in range(len(channels) - 2, -1, -1)
            ]
        )

        # Output head
        self.final_conv = nn.Conv2d(
            channels[0], out_channels, kernel_size=1, bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc_features = []

        # --- ENCODER ---
        x = self.init_conv(x)
        x = self.init_dw(x)
        enc_features.append(x)

        for i in range(len(self.enc_channels) - 1):
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = self.enc_dws[i](x)
            x = self.enc_pointconvs[i](x)
            if i < len(self.enc_channels) - 2:
                enc_features.append(x)

        # --- BOTTLENECK ---
        x = self.bottleneck_dw(x)

        # --- DECODER ---
        dec_idx = 0
        for i in range(len(self.enc_channels) - 2, -1, -1):
            target_h, target_w = (
                enc_features[i].shape[2],
                enc_features[i].shape[3],
            )
            x = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=True,
            )
            x = self.dec_pointconvs[dec_idx](x)

            # Direct skip concatenation + pointwise channel reduction + depthwise refinement
            x = torch.cat([x, enc_features[i]], dim=1)
            x = self.dec_fuse_pointconvs[dec_idx](x)
            x = self.dec_dws[dec_idx](x)
            dec_idx += 1

        # --- OUTPUT HEAD ---
        out = self.final_conv(x)
        out = self.sigmoid(out)
        return out


def aulunet_baseline():
    return AULUNetBaseline(channels=[5, 10, 15, 25, 35, 60])