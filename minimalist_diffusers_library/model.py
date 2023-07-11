import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
from diffusers import UNet2DModel # pip install diffusers
from torch.optim import Adam

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

@torch.no_grad()
def sample_iadb(model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

dataset = 'mnist' #'cifar'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT_FOLDER = '.'
transform_cifar = transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64), transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()]) 
transform_mnist = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root=ROOT_FOLDER,
                                        download=True, transform=transform_cifar if dataset=='cifar' else transform_mnist)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

model = get_model()
model = model.to(device)

optimizer = Adam(model.parameters(), lr=1e-4)
nb_iter = 0
print('Start training')

for current_epoch in tqdm(range(1000),"Epoch"):
    avg_loss = 0
    num_items = 0

    for i, data in tqdm(enumerate(dataloader),"Data-pass"):
        x1 = (data[0].to(device)*2)-1
        x0 = torch.randn_like(x1)
        bs = x0.shape[0]

        """Putting same x0 and x1 for different alphas -- modified training pipeline """
        alpha = torch.rand(bs*3, device=device)
        alpha_1 = alpha[0:bs]
        alpha_2 = alpha[0+bs:bs+bs]
        alpha_3 = alpha[0+2*bs:bs+2*bs]

        x_alpha_1 = alpha_1.view(-1,1,1,1) * x1 + (1-alpha_1).view(-1,1,1,1) * x0
        x_alpha_2 = alpha_2.view(-1,1,1,1) * x1 + (1-alpha_2).view(-1,1,1,1) * x0
        x_alpha_3 = alpha_3.view(-1,1,1,1) * x1 + (1-alpha_3).view(-1,1,1,1) * x0


        for i in range(bs):
            x_alpha = torch.concatenate([x_alpha_1[i][None, :,:,:], x_alpha_2[i][None,:,:,:], x_alpha_3[i][None,:,:,:]], dim=0)
            alpha = torch.concatenate([alpha_1[i][None,],alpha_2[i][None,],alpha_3[i][None,]])
            d = model(x_alpha, alpha)['sample']
            out = d
            x1_reformed = torch.concatenate([x1[i][None, :,:,:],x1[i][None, :,:,:],x1[i][None, :,:,:]],dim=0)
            x0_reformed = torch.concatenate([x0[i][None, :,:,:],x0[i][None, :,:,:],x0[i][None, :,:,:]],dim=0)
            loss = torch.sum((d - (x1_reformed-x0_reformed))**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        nb_iter += 1
        num_items +=1
 
        """Uncomment these lines for normal training pipeline and comment the ones above -- refer authors code at https://github.com/tchambon/IADB/blob/main/iadb.py"""

        # alpha = torch.rand(bs, device=device)
        # x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0
        
        # d = model(x_alpha, alpha)['sample']
        # loss = torch.sum((d - (x1-x0))**2)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # nb_iter += 1
        # num_items +=1
        # avg_loss += loss.item()

        if nb_iter % 200 == 0:
            with torch.no_grad():
                print(f'Save export {nb_iter}')
                sample = (sample_iadb(model, x0, nb_step=128) * 0.5) + 0.5
                torchvision.utils.save_image(sample, f'/home/rajan/Desktop/rajan_thesis/thesis_work/minimalist_diffusers_library/{dataset}_results/{dataset}_samples/export_{str(nb_iter).zfill(8)}.png')
                torch.save(model.state_dict(), f'/home/rajan/Desktop/rajan_thesis/thesis_work/minimalist_diffusers_library/{dataset}_results/{dataset}_ckpts/{dataset}.ckpt')

    avg_loss /= num_items*64
    print(f'Loss at end of {current_epoch} : {avg_loss}')
    
