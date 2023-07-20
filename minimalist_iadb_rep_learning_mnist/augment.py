import torch 
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,channels=[32, 64, 128, 256],latent_dims=2):
        super().__init__()
        self.latent_dims = latent_dims
        self.img_size = 32
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
        self.dense4 = nn.Linear(128,self.latent_dims)
        self.dense5 = nn.Linear(128,self.latent_dims)
        self.kl = 0
        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()

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
        mu = self.dense4(h7) #mu
        # h8 = self.act(h8) #the latent space
        sigma = torch.exp(self.dense5(h7) )#sigma
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = torch.sum(((sigma**2 + mu**2)/2 - torch.log(sigma) + torch.log(torch.tensor(2.0, device='cuda'))- 1/2))
        return z


class Decoder(nn.Module):
    def __init__(self, up_layers = [128,256,512,1024],latent_dims=2):
        super().__init__()
        self.up_dense1 = nn.Linear(latent_dims, up_layers[0])
        self.up_dense2 = nn.Linear(up_layers[0], up_layers[1])
        self.up_dense3 = nn.Linear(up_layers[1],up_layers[2])
        self.up_dense4 = nn.Linear(up_layers[2], up_layers[3])

        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        h9 = self.up_dense1(x)
        h9 = self.act(h9)
        h10 = self.up_dense2(h9)
        h10 = self.act(h10)
        h11 = self.up_dense3(h10)
        h11 = self.act(h11)
        h12 = self.up_dense4(h11)
        h12 = self.act(h12)   

        return h12


class VariationalEncoder(nn.Module):
    def __init__(self,latent_dims = 2):
        super().__init__()
        self.encoder = Encoder(latent_dims=latent_dims)
        self.decoder = Decoder(latent_dims=latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# model = VariationalEncoder().to('cuda')
# x = torch.randn(64,1,32,32).to('cuda')
# o = model(x)