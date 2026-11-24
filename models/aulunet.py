import torch
import torch.nn as nn
import torch.nn.functional as F

from models.akf import AdaptiveKernelFusion
from models.rgf import ResidualGlobalFusion
from models.lsg import LightweightSkipGate


DEFAULT_CHANNELS = (5, 10, 15, 25, 35, 60)


class PointConv(nn.Module):
    """PointConv block: 1x1 Conv -> BatchNorm2d -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super(PointConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Add(nn.Module):
    """Add decoder and encoder skip features."""
    def __init__(self, channels_decoder, channels_encoder):
        super(Add, self).__init__()
        if channels_decoder != channels_encoder:
            self.channel_adapter = nn.Conv2d(channels_encoder, channels_decoder, kernel_size=1, bias=True)
        else:
            self.channel_adapter = nn.Identity()

    def forward(self, decoder_features, encoder_features):
        encoder_features = self.channel_adapter(encoder_features)
        return decoder_features + encoder_features


class AULUNet(nn.Module):
    def __init__(
        self,
        channels=DEFAULT_CHANNELS,
        akf=True,
        rgf=True,
        lsg=True,
        in_channels=3,
        out_channels=1,
    ):
        super(AULUNet, self).__init__()

        self.enc_channels = list(channels)

        # Initial encoder convolution + AKF (or depthwise 1x1 fallback)
        self.init_conv = nn.Conv2d(in_channels, self.enc_channels[0], kernel_size=3, padding=1, bias=False)
        self.init_akf = self._build_akf(self.enc_channels[0], akf)

        # Encoder stages: AvgPool2d(2) -> AKF -> PointConv
        self.enc_akfs = nn.ModuleList([
            self._build_akf(self.enc_channels[i], akf)
            for i in range(len(self.enc_channels) - 1)
        ])

        self.enc_pointconvs = nn.ModuleList([
            PointConv(self.enc_channels[i], self.enc_channels[i + 1])
            for i in range(len(self.enc_channels) - 1)
        ])

        # Bottleneck RGF module (identity pass-through when rgf=False)
        self.rgf = ResidualGlobalFusion(self.enc_channels[-1]) if rgf else nn.Identity()

        # Decoder & Skip Gates (LSG or Add)
        self.dec_pointconvs = nn.ModuleList([
            PointConv(self.enc_channels[i + 1], self.enc_channels[i])
            for i in range(len(self.enc_channels) - 2, -1, -1)
        ])

        self.lsgs = nn.ModuleList([
            LightweightSkipGate(
                channels_decoder=self.enc_channels[i],
                channels_encoder=self.enc_channels[i],
            )
            if lsg
            else Add(
                channels_decoder=self.enc_channels[i],
                channels_encoder=self.enc_channels[i],
            )
            for i in range(len(self.enc_channels) - 2, -1, -1)
        ])

        self.dec_akfs = nn.ModuleList([
            self._build_akf(self.enc_channels[i], akf)
            for i in range(len(self.enc_channels) - 2, -1, -1)
        ])

        # Final output head
        self.final_conv = nn.Conv2d(self.enc_channels[0], out_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    
    def _build_akf(self, num_channels, akf_enabled):
        if akf_enabled:
            return AdaptiveKernelFusion(num_channels)
        return nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, groups=num_channels, bias=True),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        enc_features = []

        # --- ENCODER ---
        x = self.init_conv(x)
        x = self.init_akf(x)
        enc_features.append(x)

        for i in range(len(self.enc_channels) - 1):
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = self.enc_akfs[i](x)
            x = self.enc_pointconvs[i](x)
            if i < len(self.enc_channels) - 2:
                enc_features.append(x)

        # --- BOTTLENECK ---
        x = self.rgf(x)

        # --- DECODER ---
        # enc_features holds [Stage0, Stage1, Stage2, Stage3, Stage4]
        dec_idx = 0
        for i in range(len(self.enc_channels) - 2, -1, -1):
            target_h, target_w = enc_features[i].shape[2], enc_features[i].shape[3]
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)

            x = self.dec_pointconvs[dec_idx](x)
            x = self.lsgs[dec_idx](x, enc_features[i])
            x = self.dec_akfs[dec_idx](x)

            dec_idx += 1

        # --- OUTPUT HEAD ---
        out = self.final_conv(x)
        out = self.sigmoid(out)
        return out


# Factory functions for model variants specified in the paper
def baseline():
    return AULUNet(channels=[5, 10, 15, 25, 35, 60], akf=False, rgf=False, lsg=False)

def m1():
    return AULUNet(channels=[5, 10, 15, 25, 35, 60], akf=True, rgf=False, lsg=False)

def m2():
    return AULUNet(channels=[5, 10, 15, 25, 35, 60], akf=False, rgf=True, lsg=False)

def m3():
    return AULUNet(channels=[5, 10, 15, 25, 35, 60], akf=False, rgf=False, lsg=True)

def m4():
    return AULUNet(channels=[5, 10, 15, 25, 35, 60], akf=True, rgf=True, lsg=False)

def m5():
    return AULUNet(channels=[5, 10, 15, 25, 35, 60], akf=True, rgf=False, lsg=True)

def m6():
    return AULUNet(channels=[5, 10, 15, 25, 35, 60], akf=False, rgf=True, lsg=True)

def aulunet_v1():
    return AULUNet(channels=[4, 8, 16, 24, 36, 60])

def aulunet_v2():
    return AULUNet(channels=[4, 8, 16, 24, 40, 60])

def aulunet_v3():
    return AULUNet(channels=[5, 10, 15, 25, 35, 60])