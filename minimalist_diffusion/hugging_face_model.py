import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.optim import Adam
from minimalist_diff_rl_model import minimalDiffRl

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

    feats = torch.sigmoid(o1 + o2).unsqueeze(-1).unsqueeze(-1)
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
    feats = torch.sigmoid(feats)
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


class batch_norm_relu(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.bnorm = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.relu(self.bnorm(x))


class resBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()

    """Convolutional layer"""
    self.bnorm1 = batch_norm_relu(in_channels)
    self.conv1_residual = nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1,stride=stride)
    self.bnorm2 = batch_norm_relu(out_channels)
    self.conv2_residual = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1)

    """Skip connection"""
    self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)

  def forward(self, x):
    identity = x

    x = self.bnorm1(x)
    x = self.conv1_residual(x)
    x = self.bnorm2(x)
    x = self.conv2_residual(x)

    s = self.skip_connection(identity)

    return x+s


class IADBscratch(nn.Module):
  def __init__(self,in_channels, channels=[32,64,128,256]):
    super().__init__()
    
    """Convolutional Block attention modules"""
    self.pretrained_model = minimalDiffRl()
    self.cbam0 = CBAM(channels[0])
    self.cbam1 = CBAM(channels[1])
    self.cbam2 = CBAM(channels[2])
    self.cbam_bottleneck = CBAM(channels[3])
    self.cbam3 = CBAM(channels[2])
    self.cbam4 = CBAM(channels[1])

    """Skip connections for downsampling"""
    # self.skip_conn1 = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
    self.skip_conn2 = nn.Conv2d(channels[0],channels[1],kernel_size=3,stride=2)
    self.skip_conn3 = nn.Conv2d(channels[1],channels[2],kernel_size=3,stride=2)
    self.skip_conn4 = nn.Conv2d(channels[2],channels[3],kernel_size=3,stride=2)

    """Skip connections for upsampling"""
    self.skip_conn5 = nn.Conv2d(channels[3],channels[2], kernel_size=3, stride=2)
    self.skip_conn6 = nn.Conv2d(channels[2],channels[1], kernel_size=3, stride=2)
    self.skip_conn7 = nn.Conv2d(channels[1],channels[0], kernel_size=3, stride=2)

    """Extra convolution operations downsampling"""
    # self.extra_conv1 = nn.Conv2d(channels[0],channels[0],kernel_size=3,padding=1,stride=1)
    self.extra_conv2 = nn.Conv2d(channels[1],channels[1],kernel_size=3,padding=1,stride=1)
    self.extra_conv3 = nn.Conv2d(channels[2],channels[2],kernel_size=3,padding=1,stride=1)
    self.extra_conv4 = nn.Conv2d(channels[3],channels[3],kernel_size=3,padding=1,stride=1)


    """Extra convolution operations upsampling"""



  def forward(self, x,alpha):
    embed = self.pretrained_model.embed(alpha)

    #downsampling
    h1 = self.pretrained_model.encoder_drl.conv1(x)
    h1 += self.pretrained_model.dense_embed1(embed)
    h1 = self.pretrained_model.encoder_drl.gnorm1(h1)
    h1 = self.pretrained_model.encoder_drl.act_relu(h1)
    h1 = self.cbam0(h1)


    h2 = self.pretrained_model.encoder_drl.conv2(h1)
    h2_identity = h1
    h2 = self.pretrained_model.encoder_drl.gnorm2(h2)
    h2 = self.pretrained_model.encoder_drl.act_relu(h2)
    h2 = self.extra_conv2(h2)
    h2_skip = self.skip_conn2(h2_identity)
    h2 += h2_skip
    h2 += self.pretrained_model.dense_embed2(embed)
    h2 = self.pretrained_model.encoder_drl.gnorm2(h2)
    h2 = self.pretrained_model.encoder_drl.act_relu(h2)
    h2 = self.cbam1(h2)



    h3 = self.pretrained_model.encoder_drl.conv3(h2)
    h3_identity = h2
    h3 = self.pretrained_model.encoder_drl.gnorm3(h3)
    h3 = self.pretrained_model.encoder_drl.act_relu(h3)
    h3 = self.extra_conv3(h3)
    h3_skip = self.skip_conn3(h3_identity)
    h3 += h3_skip
    h3 += self.pretrained_model.dense_embed3(embed)
    h3 = self.pretrained_model.encoder_drl.gnorm3(h3)
    h3 = self.pretrained_model.encoder_drl.act_relu(h3)
    h3 = self.cbam2(h3)


    h4 = self.pretrained_model.encoder_drl.conv4(h3)
    h4_identity = h3
    h4 = self.pretrained_model.encoder_drl.gnorm4(h4)
    h4 = self.pretrained_model.encoder_drl.act_relu(h4)
    h4 = self.extra_conv4(h4)
    h4_skip = self.skip_conn4(h4_identity)
    h4 += h4_skip
    h4 += self.pretrained_model.dense_embed4(embed)
    h4 = self.pretrained_model.encoder_drl.gnorm4(h4)
    h4 = self.pretrained_model.encoder_drl.act_relu(h4)
    h4 = self.cbam_bottleneck(h4)


    #Upsampling
    h = self.pretrained_model.encoder_drl.upconv1(h4)
    h+= self.pretrained_model.dense_embed5(embed)
    h = self.pretrained_model.encoder_drl.gnorm5(h)
    h = self.pretrained_model.encoder_drl.act_relu(h)
    h = self.cbam3(h)

    h = self.pretrained_model.encoder_drl.upconv2(torch.cat([h,h3],dim=1))
    h+= self.pretrained_model.dense_embed6(embed)
    h = self.pretrained_model.encoder_drl.gnorm6(h)
    h = self.pretrained_model.encoder_drl.act_relu(h)
    h = self.cbam4(h)

    h = self.pretrained_model.encoder_drl.upconv3(torch.cat([h,h2],dim=1))
    h+= self.pretrained_model.dense_embed7(embed)
    h = self.pretrained_model.encoder_drl.gnorm7(h)
    h = self.pretrained_model.encoder_drl.act_relu(h)

    h = self.pretrained_model.encoder_drl.upconv4(torch.cat([h,h1],dim=1))
    h = self.pretrained_model.encoder_drl.act_swish(h)

    
    return h



    
class IADB(nn.Module):
  def __init__(self,in_channels, channels=[32,64,128,256]):
    super().__init__()
    
    """Convolutional Block attention modules"""
    self.pretrained_model = minimalDiffRl()
    self.cbam1 = CBAM(channels[1])
    self.cbam2 = CBAM(channels[2])
    self.cbam3 = CBAM(channels[2])
    self.cbam4 = CBAM(channels[1])

    """Skip connections for downsampling"""
    # self.skip_conn1 = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
    self.skip_conn2 = nn.Conv2d(channels[0],channels[1],kernel_size=3,stride=2)
    self.skip_conn3 = nn.Conv2d(channels[1],channels[2],kernel_size=3,stride=2)
    self.skip_conn4 = nn.Conv2d(channels[2],channels[3],kernel_size=3,stride=2)

    """Skip connections for upsampling"""
    self.skip_conn5 = nn.Conv2d(channels[3],channels[2], kernel_size=3, stride=2)
    self.skip_conn6 = nn.Conv2d(channels[2],channels[1], kernel_size=3, stride=2)
    self.skip_conn7 = nn.Conv2d(channels[1],channels[0], kernel_size=3, stride=2)

    """Extra convolution operations downsampling"""
    # self.extra_conv1 = nn.Conv2d(channels[0],channels[0],kernel_size=3,padding=1,stride=1)
    self.extra_conv2 = nn.Conv2d(channels[1],channels[1],kernel_size=3,padding=1,stride=1)
    self.extra_conv3 = nn.Conv2d(channels[2],channels[2],kernel_size=3,padding=1,stride=1)
    self.extra_conv4 = nn.Conv2d(channels[3],channels[3],kernel_size=3,padding=1,stride=1)


    """Extra convolution operations upsampling"""



  def forward(self, x,alpha):
    embed = self.pretrained_model.embed(alpha)

    #downsampling
    h1 = self.pretrained_model.encoder_drl.conv1(x)
    h1 += self.pretrained_model.dense_embed1(embed)
    h1 = self.pretrained_model.encoder_drl.gnorm1(h1)
    h1 = self.pretrained_model.encoder_drl.act_relu(h1)


    h2 = self.pretrained_model.encoder_drl.conv2(h1)
    h2_identity = h1
    h2 = self.pretrained_model.encoder_drl.gnorm2(h2)
    h2 = self.pretrained_model.encoder_drl.act_relu(h2)
    h2 = self.extra_conv2(h2)
    h2_skip = self.skip_conn2(h2_identity)
    h2 += h2_skip
    h2 += self.pretrained_model.dense_embed2(embed)
    h2 = self.pretrained_model.encoder_drl.gnorm2(h2)
    h2 = self.pretrained_model.encoder_drl.act_relu(h2)
    h2 = self.cbam1(h2)



    h3 = self.pretrained_model.encoder_drl.conv3(h2)
    h3_identity = h2
    h3 = self.pretrained_model.encoder_drl.gnorm3(h3)
    h3 = self.pretrained_model.encoder_drl.act_relu(h3)
    h3 = self.extra_conv3(h3)
    h3_skip = self.skip_conn3(h3_identity)
    h3 += h3_skip
    h3 += self.pretrained_model.dense_embed3(embed)
    h3 = self.pretrained_model.encoder_drl.gnorm3(h3)
    h3 = self.pretrained_model.encoder_drl.act_relu(h3)
    h3 = self.cbam2(h3)


    h4 = self.pretrained_model.encoder_drl.conv4(h3)
    h4_identity = h3
    h4 = self.pretrained_model.encoder_drl.gnorm4(h4)
    h4 = self.pretrained_model.encoder_drl.act_relu(h4)
    h4 = self.extra_conv4(h4)
    h4_skip = self.skip_conn4(h4_identity)
    h4 += h4_skip
    h4 += self.pretrained_model.dense_embed4(embed)
    h4 = self.pretrained_model.encoder_drl.gnorm4(h4)
    h4 = self.pretrained_model.encoder_drl.act_relu(h4)


    #Upsampling
    h = self.pretrained_model.encoder_drl.upconv1(h4)
    h+= self.pretrained_model.dense_embed5(embed)
    h = self.pretrained_model.encoder_drl.gnorm5(h)
    h = self.pretrained_model.encoder_drl.act_relu(h)
    h = self.cbam3(h)

    h = self.pretrained_model.encoder_drl.upconv2(torch.cat([h,h3],dim=1))
    h+= self.pretrained_model.dense_embed6(embed)
    h = self.pretrained_model.encoder_drl.gnorm6(h)
    h = self.pretrained_model.encoder_drl.act_relu(h)
    h = self.cbam4(h)

    h = self.pretrained_model.encoder_drl.upconv3(torch.cat([h,h2],dim=1))
    h+= self.pretrained_model.dense_embed7(embed)
    h = self.pretrained_model.encoder_drl.gnorm7(h)
    h = self.pretrained_model.encoder_drl.act_relu(h)

    h = self.pretrained_model.encoder_drl.upconv4(torch.cat([h,h1],dim=1))
    h = self.pretrained_model.encoder_drl.act_swish(h)

    
    return h
    
    
# y = torch.randn(32)
# x = torch.randn(32,1,28,28)
# model = IADBscratch(in_channels=1)
# o = model(x,y )
