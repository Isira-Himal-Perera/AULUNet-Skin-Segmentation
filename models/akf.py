import torch
import torch.nn as nn

class AdaptiveKernelFusion(nn.Module):
    def __init__(self, channels):
        super(AdaptiveKernelFusion, self).__init__()
        
        # Local path: 3x3 depthwise convolution with dilation = 1
        self.local_path = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels, bias=False
        )
        
        # Dilated path: 3x3 depthwise convolution with dilation = 2
        self.dilated_path = nn.Conv2d(
            channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False
        )
        
        # Gating mechanism (GAP -> 1x1 Conv -> Sigmoid)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        # Post-fusion projection block: 1x1 Conv -> BatchNorm -> ReLU
        self.pointwise_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. Parallel feature extraction paths
        F_l = self.local_path(x)
        F_g = self.dilated_path(x)
        
        # 2. Dynamic gating weights alpha
        gate_feat = self.gap(x)
        gate_feat = self.gate_conv(gate_feat)
        alpha = self.sigmoid(gate_feat) # Shape: [B, C, 1, 1]
        
        # 3. Adaptive fusion: y = alpha * F_l + (1 - alpha) * F_g
        y = alpha * F_l + (1.0 - alpha) * F_g
        
        # 4. Final projection, batch norm, and ReLU
        z = self.pointwise_fusion(y)
        return z