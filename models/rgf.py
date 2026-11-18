import torch
import torch.nn as nn

class ResidualGlobalFusion(nn.Module):
    def __init__(self, channels):
        super(ResidualGlobalFusion, self).__init__()
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 1x1 Convolution to generate modulation coefficients
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        
        # Tanh activation to scale coefficients between [-1, 1]
        self.tanh = nn.Tanh()

    def forward(self, B):
        # 1. Compute global descriptor via GAP
        global_desc = self.gap(B)
        
        # 2. Transform through 1x1 conv and tanh activation -> G in [-1, 1]
        G = self.conv1x1(global_desc)
        G = self.tanh(G)
        
        # 3. Residual modulation: B' = B + (B * G)
        B_prime = B + (B * G)
        
        return B_prime
    