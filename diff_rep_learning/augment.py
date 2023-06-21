import torch 
import torch.nn as nn 
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self,channels=[32, 64, 128, 256]):
        super().__init__()
        self.img_size = 28
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        ## Paper implementation -- three additional dense layers mapping it to 
        ## the required latent dimension space 
        self.flatten = nn.Flatten()
        
        #downsampling block -- latent space 2D 
        self.dense1 = nn.Linear(1024,512)
        self.dense2 = nn.Linear(512,256)
        self.dense3 = nn.Linear(256,128)
        self.dense4 = nn.Linear(128,2)
        
        #upsampling block -- coding it to higher dimensional space
        self.up_dense1 = nn.Linear(2,128)
        self.up_dense2 = nn.Linear(128,256)
        self.up_dense3 = nn.Linear(256,512)
        self.up_dense4 = nn.Linear(512,1024)

        #actiavtions 
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        h1 = self.conv1(x)
        h1 = self.gnorm1(h1)
        h1 = F.relu(h1)
        h2 = self.conv2(h1)
        h2 = self.gnorm2(h2)
        h2 = F.relu(h2)
        h3 = self.conv3(h2)
        h3 = self.gnorm3(h3)
        h3 = F.relu(h3)
        h4 = self.conv4(h3)
        h4 = self.gnorm4(h4)
        h4 = F.relu(h4)
        #Flattening the output
        h5 = self.flatten(h4)
        h5 = self.dense1(h5)
        h5 = self.act(h5)
        h6 = self.dense2(h5)
        h6 = self.act(h6)
        h7 = self.dense3(h6)
        h7 = self.act(h7)
        h8 = self.dense4(h7)
        h8 = self.act(h8) #the latent space

        #encoding it to a highert dimensional space 
        h9 = self.up_dense1(h8)
        h9 = self.act(h9)
        h10 = self.up_dense2(h9)
        h10 = self.act(h10)
        h11 = self.up_dense3(h10)
        h11 = self.act(h11)
        h12 = self.up_dense4(h11)
        h12 = self.act(h12)   

        return h12



# class Decoder(nn.Module):
#     def __init__(self,channels=[32, 64, 128, 256], embed_dim=256):
#         super().__init__()
        

#     def forward(self, x):
#         pass



# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         pass

#     def forward(self, x):
#         pass

x = torch.randn(32,1,28,28)
model = Encoder()

output = model(x)
