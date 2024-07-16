import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import LayerNorm2d

class MaskFeatureBlock(nn.Module):
    def __init__(self, transformer_dim):
        super(MaskFeatureBlock, self).__init__()        
        self.dnc_block_combined =  DNCBlock_combined(transformer_dim // 8) 
        self.fgm_block = FGMBlock(transformer_dim // 8)  
        self.conv_layer = nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, padding=1)        

        self.downsample_layer = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1)
        )

    def forward(self, x, clear=True):
        if not clear:
            x = self.dnc_block_combined(x)
            x = self.fgm_block(x)                
            x = self.conv_layer(x)   
            
        output = self.downsample_layer(x)
        return output

class FirstLayerFeatureBlock(nn.Module):
    def __init__(self, vit_dim, transformer_dim):
        super(FirstLayerFeatureBlock, self).__init__()
        self.dnc_block_combined =  DNCBlock_combined(vit_dim) 
        self.fgm_block = FGMBlock(vit_dim)  
        self.conv_layer = nn.Conv2d(2*vit_dim, vit_dim, kernel_size=3, padding=1)

        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2)
        )

    def forward(self, x, clear=True):
        if not clear:
            x = self.dnc_block_combined(x)
            x = self.fgm_block(x)                
            x = self.conv_layer(x)   

        output = self.upsample_layer(x)
        return output

class LastLayerFeatureBlock(nn.Module):
    def __init__(self, transformer_dim):
        super(LastLayerFeatureBlock, self).__init__()
        self.dnc_block_combined =  DNCBlock_combined(transformer_dim) 
        self.fgm_block = FGMBlock(transformer_dim)  
        self.conv_layer = nn.Conv2d(2*transformer_dim, transformer_dim, kernel_size=3, padding=1)             
        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2)
        )

    def forward(self, x, clear=True):
        if not clear:
            x = self.dnc_block_combined(x)
            x = self.fgm_block(x)                
            x = self.conv_layer(x)   

        output = self.upsample_layer(x)
        return output

class TokenBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super(TokenBlock, self).__init__()
        self.input_dim = input_dim

        # breakpoint()
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim * input_dim)
        )
        
        self.IN_layer_I = nn.InstanceNorm1d(input_dim)
        self.IN_layer_II = nn.InstanceNorm1d(input_dim)

    def forward(self, x, mlp=True):
        x = self.IN_layer_I(x)
        x = self.IN_layer_II(x)
        x = x.view(self.input_dim, -1)
        output = self.mlp(x)

        return output

class DNCBlock_combined(nn.Module):
    def __init__(self, vit_dim):
        super(DNCBlock_combined, self).__init__()
        self.num_channels = vit_dim
        self.channel_attention = CABlock(2*self.num_channels)
        self.SEMBlock = SKDown(3, 1, False, 16, self.num_channels, self.num_channels, first=False)

    def forward(self, x):
        x_in = self.SEMBlock(x)
        x_all = torch.cat([x, x_in], dim=1)
        output = self.channel_attention(x_all)
        
        return output   
    
class Selector(nn.Module):
    def __init__(self, channel, reduction=16, crp_classify=False):
        super(Selector, self).__init__()
        self.spatial_attention = 4
        self.in_channel = channel * (self.spatial_attention ** 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((self.spatial_attention, self.spatial_attention))

        self.fc = nn.Sequential(
            nn.Linear(self.in_channel, self.in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.att_conv1 = nn.Linear(self.in_channel // reduction, self.in_channel)
        self.att_conv2 = nn.Linear(self.in_channel // reduction, self.in_channel)

    def forward(self, x):

        b, c, H, W = x.size()

        y = self.avg_pool(x).reshape(b, -1)
        y = self.fc(y)

        att1 = self.att_conv1(y).view(b, c, self.spatial_attention, self.spatial_attention)
        att2 = self.att_conv2(y).view(b, c, self.spatial_attention, self.spatial_attention)

        attention = torch.stack((att1, att2))
        attention = nn.Softmax(dim=0)(attention)

        att1 = F.interpolate(attention[0], scale_factor=(H / self.spatial_attention, W / self.spatial_attention), mode="nearest")
        att2 = F.interpolate(attention[1], scale_factor=(H / self.spatial_attention, W / self.spatial_attention), mode="nearest")

        return att1, att2

class SelectiveConv(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels, first=False):
        super(SelectiveConv, self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.selector = Selector(out_channels, reduction=reduction)
        self.IN = nn.InstanceNorm2d(in_channels)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):    
        if self.first:
            f_input = x
            s_input = x
        else:
            f_input = self.BN(x.clone())
            f_input = self.relu(f_input)

            s_input = self.IN(x.clone())
            s_input = self.relu(s_input)

        out1 = self.conv1(f_input)
        out2 = self.conv2(s_input)

        out = out1 + out2

        att1, att2 = self.selector(out)
        out = torch.mul(out1, att1) + torch.mul(out2, att2)

        return out

class SKDown(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels, first=False):
        super(SKDown, self).__init__()
        self.maxpool_conv = nn.Sequential(
            SelectiveConv(kernel_size, padding, bias, reduction, in_channels, out_channels, first=first)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class CABlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CABlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeeze = self.squeeze(x).view(batch_size, channels)
        excitation = self.excitation(squeeze).view(batch_size, channels, 1, 1)
        return x * excitation
    
class FGMBlock(nn.Module):
    def __init__(self, vit_dim):
        super(FGMBlock, self).__init__()
        self.num_channels = 2 * vit_dim
        self.conv_layer = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=1)

    def forward(self, x):
        fft_map = torch.fft.fft2(x, dim=(-2, -1))
        
        magnitude_map = torch.abs(fft_map)
        phase_map = torch.angle(fft_map)
        
        modified_magnitude = self.conv_layer(magnitude_map)
        
        real_part = modified_magnitude * torch.cos(phase_map)
        imag_part = modified_magnitude * torch.sin(phase_map)
        modified_fft_map = torch.complex(real_part, imag_part)

        reconstructed_x = torch.real(torch.fft.ifft2(modified_fft_map, dim=(-2, -1)))
        
        return reconstructed_x

