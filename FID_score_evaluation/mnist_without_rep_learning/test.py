"""
WITHOUT REPRESENTATION LEANRING 

"""
import torchvision
import torch
from model import Unet,sample_IADB
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
dataset = torchvision.datasets.MNIST('..', train=True, download=False,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ]))
batchsize = 64
torch.manual_seed(0)
dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=4, drop_last=True, shuffle=True)   

real_images,fake_images = None, None

def preprocess_images(image: torch.tensor):
    mnist = -1 + 2*image.to("cpu")
    mnist = torch.nn.functional.interpolate(mnist, size=(32,32), mode='bilinear', align_corners=False)
    return mnist
        
i,(x, y) = next(enumerate(dataloader))
real_images = preprocess_images(x)

real_images = real_images.expand(-1,3,-1,-1)
print(real_images.shape)

#model loading

D = Unet()
epoch = 39
gamma = 0
ckpt_d = D.load_state_dict(torch.load(f'/home/rajan/Desktop/rajan_thesis/thesis_work/FID_score_evaluation/mnist_without_rep_learning/checkpoints/d_epoch{epoch}.ckpt'))
D = D.to('cpu')

fake_images = sample_IADB(D,'test',device='cpu')
fake_images = fake_images.expand(-1,3,-1,-1)
print(fake_images.shape)

real_images = real_images.to('cpu')
fake_images = fake_images.to('cpu')
fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"FID without representation learning: {float(fid.compute())}")