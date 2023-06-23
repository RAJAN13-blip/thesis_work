import torch 
import torch.nn.functional as F
import torch.nn as nn
from minimalist_diff_model import DiffusionNet
# from ..diff_rep_learning.augment import Encoder

class minimalDiffRl_revised(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DiffusionNet()


    def forward(self, x, alpha):
        """x :blended images
           alpha :pixelated alpha information
        """
        #downsampling  block 
        h1 = self.model.conv1(x)
        h1_alpha = self.model.conv1(alpha)
        h1 += h1_alpha
        h1 = self.model.gnorm1(h1)
        h1 = self.model.act_relu(h1)

        h2 = self.model.conv2(h1)
        h2_alpha = self.model.conv2(self.model.act_relu(self.model.gnorm1(h1_alpha)))
        h2 += h2_alpha
        h2 = self.model.gnorm2(h2)
        h2 = self.model.act_relu(h2)

        h3 = self.model.conv3(h2)
        h3_alpha = self.model.conv3(self.model.act_relu(self.model.gnorm2(h2_alpha)))
        h3 += h3_alpha
        h3 = self.model.gnorm3(h3)
        h3 = self.model.act_relu(h3)

        h4 = self.model.conv4(h3)
        h4_alpha = self.model.conv4(self.model.act_relu(self.model.gnorm3(h3_alpha)))
        h4 += h4_alpha
        h4 = self.model.gnorm4(h4)
        h4 = self.model.act_relu(h4)

        #upsampling with skip connection
        h = self.model.upconv1(h4)
        h1_up_alpha = self.model.upconv1(h4_alpha)
        h += h1_up_alpha
        h = self.model.gnorm5(h)
        h = self.model.act_relu(h)

        h = self.model.upconv2(torch.cat([h,h3],dim=1))
        h2_up_alpha = self.model.upconv2(torch.cat([h3_alpha,h1_up_alpha],dim=1))
        h += h2_up_alpha
        h = self.model.gnorm6(h)
        h = self.model.act_relu(h)

        h = self.model.upconv3(torch.cat([h,h2],dim=1))
        h3_up_alpha = self.model.upconv3(torch.cat([h2_alpha,h2_up_alpha],dim=1))
        h += h3_up_alpha
        h = self.model.gnorm7(h)
        h = self.model.act_relu(h)

        h = self.model.upconv4(torch.cat([h,h1],dim=1))
        h = self.model.act_swish(h)

        return h



# class minimalDiffRl_with_encoder(nn.Module):
#     def __init__(self, pretrain=True):
#         super().__init__()
#         self.minimal_model = minimalDiffRl_revised()
#         self.rep_encoder = Encoder()


#     def forward():

#         pass