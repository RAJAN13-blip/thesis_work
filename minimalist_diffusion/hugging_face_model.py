import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam
from torchinfo import summary


device = 'cpu'
def get_model():
    block_out_channels=(128, 128, 256, 256, 512, 512)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=1, in_channels=1, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)


class channel_attention(nn.Module):
  def __init__(self, in_channels, channel_ratio=8):
    super().__init__()

    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)

    self.shared_mlp = nn.Sequential(
        nn.Linear(in_channels, in_channels//channel_ratio, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(in_channels//channel_ratio, in_channels, bias=False)
    )


  def forward(self, x):
    x1 = self.avg_pool(x).squeeze(-1).squeeze(-1)
    x2 = self.max_pool(x).squeeze(-1).squeeze(-1)

    o1 = self.shared_mlp(x1)
    o2 = self.shared_mlp(x2)

    feats = F.sigmoid(o1 + o2).unsqueeze(-1).unsqueeze(-1)
    h = x* feats
    return h
 

class spatial_attention(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv = nn.Conv2d(2,1,kernel_size=7,padding=3,bias=False)

  def forward(self, x):
    x1 = torch.mean(x,dim=1,keepdim=True)
    x2 , _ = torch.max(x,dim=1,keepdim=True)

    feats = torch.cat([x1,x2],dim=1)
    feats = self.conv(feats)
    feats = F.sigmoid(feats)
    refined_feats = x * feats

    return refined_feats



class CBAM(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels
    self.sam = spatial_attention()
    self.cam = channel_attention(self.in_channels)


  def forward(self, x):
    cam = self.cam(x)
    sam = self.sam(cam)

    return sam


