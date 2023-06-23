import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
import os 
import torch.nn as nn


def dataset_loader():
    """
    """
    pass

def train_model(model: nn.Module,
                data : DataLoader,
                last_epoch : int,
                num_epochs : int,
                model_path : str) -> None:
    """train the given model for given number of steps

    Args:
        model (nn.Module): 
        data (DataLoader): 
        last_epoch (int) : 
        num_epochs (int) : 
        model_path (str) : 
    """
    pass

def plot_tensor(x: torch.tensor) -> None:
    """plotting or visualizing a tensor with PIL 

    Args:
        x (torch.tensor): a single tensor
    """
    assert(len(x.shape)==4 and x.shape[0]==1)
    copy_image = x[0]
    copy_image = torch.permute(copy_image,(1,2,0))
    plt.imshow(copy_image)
    return copy_image


def plot_grid_images():
    """
    """
    pass

def get_model_to_train():
    """
    """
    pass


def get_model_to_test():
    """
    """
    pass