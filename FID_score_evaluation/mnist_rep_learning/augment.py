import torch 
import torch.nn as nn 
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 in_channels = 1,  
                 channels = [32, 64, 128, 256], 
                 down_samples = [1024, 512, 256, 128],
                 latent_dims = 2):
        super().__init__()
        self.latent_dims = latent_dims
        self.img_size = 32

        # Convolutional downsample
        self.conv1 = nn.Conv2d(in_channels,channels[0], 3, stride=1, bias=False)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)


        # Group Normalization layers 
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # For feature extraction and latent space embeddings 
        self.dense1 = nn.Linear(down_samples[0],down_samples[1])
        self.dense2 = nn.Linear(down_samples[1],down_samples[2])
        self.dense3 = nn.Linear(down_samples[2],down_samples[3])
        self.dense4 = nn.Linear(down_samples[3],self.latent_dims )
        self.dense5 = nn.Linear(down_samples[3],self.latent_dims )
        self.flatten = nn.Flatten()

        self.kl = 0
        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

        #activations 
        self.act = lambda x : x * torch.sigmoid(x)

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
        self.kl = torch.sum((2*(sigma**2 + mu**2)/2 - torch.log(sigma) + torch.log(torch.tensor(1.0, device='cuda'))- 1/2))
        return z

class Decoder(nn.Module):
    def __init__(self, 
                 out_channels = 1,
                 up_samples= [128,256,512,1024],
                 channels=[32, 64, 128, 256],
                 latent_dims=2):
        super().__init__()

        # Convolutionsal upsample layers 
        self.channels = channels
        self.latent_dims = latent_dims
        self.conv_up1 = nn.ConvTranspose2d(channels[3],channels[2], kernel_size=3, stride=2, output_padding=1,  bias=False)
        self.conv_up2 = nn.ConvTranspose2d(channels[2],channels[1], kernel_size=3, stride=2, output_padding=1, bias=False)
        self.conv_up3 = nn.ConvTranspose2d(channels[1],channels[0], kernel_size=3, stride=2, output_padding=1, bias=False)
        self.conv_up4 = nn.ConvTranspose2d(channels[0],out_channels, kernel_size=3, stride=1,bias=False)

        # Linear upsampling from latent space
        self.dense_up1 = nn.Linear(latent_dims, up_samples[0])
        self.dense_up2 = nn.Linear(up_samples[0], up_samples[1])
        self.dense_up3 = nn.Linear(up_samples[1], up_samples[2])
        self.dense_up4 = nn.Linear(up_samples[2], up_samples[3])

        #Group normalizaiton layers 
        self.gnorm_up1 = nn.GroupNorm(32,num_channels=channels[2])
        self.gnorm_up2 = nn.GroupNorm(32,num_channels=channels[1])
        self.gnorm_up3 = nn.GroupNorm(32,num_channels=channels[0])

        self.higher_latent = None
        #activation layers 
        self.act = lambda x : x*torch.sigmoid(x)
    
    def forward(self, x):
        x = self.act(self.dense_up1(x))
        x = self.act(self.dense_up2(x))
        x = self.act(self.dense_up3(x))
        x = self.act(self.dense_up4(x))

        self.higher_latent = x

        x = x.reshape(-1, self.channels[3], 2, 2)

        h1 = F.relu(self.conv_up1(x))
        h1 = self.gnorm_up1(h1)
        h2 = F.relu(self.conv_up2(h1))
        h2 = self.gnorm_up2(h2)
        h3 = F.relu(self.conv_up3(h2))
        h3 = self.gnorm_up3(h3)
        h4 = F.relu(self.conv_up4(h3))

        return h4 , self.higher_latent


class VAE(nn.Module):
    def __init__(self , in_channels = 1,latent_dims = 2):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels,latent_dims=latent_dims)
        self.decoder = Decoder(out_channels=in_channels,latent_dims=latent_dims)
        self.recon_loss = 0

    def forward(self, x):
        x_original  = x
        z = self.encoder(x)
        x_reconstructed , higher_latent =  self.decoder(z)
        self.recon_loss = ((x_original - x_reconstructed)**2).sum()

        return higher_latent

class Decoder_without_recon(nn.Module):
    def __init__(self,
                  up_layers = [128,256,512,1024],
                  latent_dims=2):
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


class VAE_without_recon(nn.Module):
    def __init__(self,in_channels=1, latent_dims = 2):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels,latent_dims=latent_dims)
        self.decoder = Decoder_without_recon(latent_dims=latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# x = torch.randn(64, 1, 32,32).to(device='cuda')
# model = Encoder()
# model.to(device='cuda')
# o = model(x)
# model_up = Decoder()
# model_up.to(device='cuda')
# z = model_up(o)