import torch
import torch.nn as nn

class LightweightSkipGate(nn.Module):
    def __init__(self, channels_decoder, channels_encoder):
        super(LightweightSkipGate, self).__init__()
        
        # 1x1 conv to ensure channel consistency if encoder and decoder channels differ
        if channels_decoder != channels_encoder:
            self.channel_adapter = nn.Conv2d(channels_encoder, channels_decoder, kernel_size=1, bias=False)
        else:
            self.channel_adapter = nn.Identity()
            
        # Gating mechanism: processes concatenated [D, E'] features
        # Input channels = channels_decoder (D) + channels_decoder (adapted E') = 2 * channels_decoder
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels_decoder * 2, channels_decoder, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_decoder, momentum=0.1),
            nn.Sigmoid()
        )

    def forward(self, D, E):
        # 1. Ensure channel consistency for encoder features
        E_prime = self.channel_adapter(E)
        
        # 2. Concatenate decoder features D and adapted encoder features E' along channel dim
        concat_feat = torch.cat([D, E_prime], dim=1)
        
        # 3. Generate spatial attention weights beta
        beta = self.gate_conv(concat_feat)
        
        # 4. Fuse via element-wise addition: S = D + beta * E'
        S = D + (beta * E_prime)
        
        return S
