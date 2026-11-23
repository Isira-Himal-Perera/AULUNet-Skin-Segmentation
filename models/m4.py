import torch
import torch.nn as nn
import torch.nn.functional as F

from models.akf import AdaptiveKernelFusion
from models.rgf import ResidualGlobalFusion

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

class AULUNetM4(nn.Module):
    def __init__(self, channels=[5, 10, 15, 25, 35, 60], in_channels=3, out_channels=1):
        super(AULUNetM4, self).__init__()
        
        # Initial encoder convolution + AKF
        self.init_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False)
        self.init_akf = AdaptiveKernelFusion(channels[0])
        
        # Encoder stages (6 stages total)
        # Stage 0 to Stage 5 channel progressions
        self.enc_channels = channels
        
        # We can build encoder stages
        # Stage 0: channels[0] -> channels[0] (handled by init)
        # Subsequent stages use AvgPool2d(2) then AKF + PointConv
        self.enc_akfs = nn.ModuleList([
            AdaptiveKernelFusion(channels[i]) 
            for i in range(len(channels)-1)
        ])
        
        self.enc_pointconvs = nn.ModuleList([
            PointConv(channels[i], channels[i+1])
            for i in range(len(channels)-1)
        ])
        
        # Bottleneck RGF module
        self.rgf = ResidualGlobalFusion(channels[-1])
        
        # Decoder & Skip Gates (LSG)
        # Decoder flows bottom-up from bottleneck back to full resolution
        self.dec_pointconvs = nn.ModuleList([
            PointConv(channels[i+1], channels[i])
            for i in range(len(channels)-2, -1, -1)
        ])

        # add decoder and encodeer channels 
        self.skip_connections = nn.ModuleList([
            Add(channels_decoder=channels[i], channels_encoder=channels[i])
            for i in range(len(channels)-2, -1, -1)
        ])
        
        self.dec_akfs = nn.ModuleList([
            AdaptiveKernelFusion(channels[i])
            for i in range(len(channels)-2, -1, -1)
        ])
        
        # Final output head
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Store encoder feature maps for skip connections
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
        # enc_features order from top to bottom: [Stage0, Stage1, Stage2, Stage3, Stage4]
        # Decoder goes backwards from Stage 4 down to Stage 0
        dec_idx = 0
        for i in range(len(self.enc_channels) - 2, -1, -1):
            # Upsample via bilinear interpolation
            target_h, target_w = enc_features[i].shape[2], enc_features[i].shape[3]
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
            
            # PointConv on decoder feature
            x = self.dec_pointconvs[dec_idx](x)
            
            # Skip connection via LSG (Decoder feature x, Encoder feature enc_features[i])
            x = self.skip_connections[dec_idx](x, enc_features[i])
            
            # AKF refinement on decoded features
            x = self.dec_akfs[dec_idx](x)
            
            dec_idx += 1
            
        # --- OUTPUT HEAD ---
        out = self.final_conv(x)
        out = self.sigmoid(out)
        return out

# Factory functions for model variants specified in the paper
def m4():
    return AULUNetM4()