import numpy as np
import matplotlib.pyplot as plt
import imageio
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import random
import cv2
import os
import re
import pandas as pd
import math
from collections import OrderedDict
import gc
import tqdm as tqdm

def distance(xy, xy1):
    dist = [(a - b)**2 for a, b in zip(xy, xy1)]
    dist = math.sqrt(sum(dist))
    return dist

def order_coords(x, y, x1, y1, sort_coords, sort_coords_asc):
    xy = pd.DataFrame({'x':x, 'y':y}).sort_values(sort_coords, ascending=sort_coords_asc)
    xy1 = pd.DataFrame({'x':x1, 'y':y1}).sort_values(sort_coords, ascending=sort_coords_asc)
    
    x = xy['x'].values.tolist()
    y = xy['y'].values.tolist()
    x1 = xy1['x'].values.tolist()
    y1 = xy1['y'].values.tolist()
    return x, y, x1, y1

def order_dist(x, y, x1, y1):
    sorted_x1 = []
    sorted_y1 = []

    for i in tqdm(range(len(x)), desc='Euclidean sort', leave=False):
        dist = 9999999
        for i in range(len(x1)):
            dist_n = distance((x[0], y[0]), (x1[i], y1[i]))
            if dist_n < dist:
                dist = dist_n
                idx = i

        sorted_x1.append(x1.pop(idx))
        sorted_y1.append(y1.pop(idx))
    x1 = sorted_x1
    y1 = sorted_y1
    return x1, y1

def draw_axis(axis_on, ax):
    if(axis_on):
        # grid
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
        ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
    else:
        # remove spines
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # remove ticks
        plt.xticks([])
        plt.yticks([])

def match_sizes(x, y, size):
    while len(x) < size:
        diff = size - len(x)
        x = x + x[:diff]
        y = y + y[:diff]

    return x, y

# transform a letter into random x/y points with the shape of that letter
def get_masked_data(letter, intensity = 2, rand=True, in_path=None):

    # get mask from image
    if in_path:
        mask = cv2.imread(os.path.join(in_path, f'{letter.upper()}.png',0))
        mask = cv2.flip(mask, 0)
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mask = pd.read_pickle(os.path.join(dir_path,'masks.pkl'))
        mask = mask[mask['letter'] == letter.upper()]['mask'].values[0]
    # fill a plot with random points
    if rand:
        random.seed(420)
        x = []
        y = []
        
        for i in range(intensity):
            x = x + random.sample(range(0, 1000), 500)
            y = y + random.sample(range(0, 1000), 500)

    # fill a plot with evenly sparced points
    else:
        base = np.arange(0, 1000, intensity, dtype=int).tolist()
        y = []
        x = []

        for i in base:
            x_temp = [i]*len(base)
            y_temp = base
            x = x + x_temp
            y = y + y_temp

    # get only the coordinates inside the mask
    result_x = []
    result_y = []

    for i in range(len(x)):
        if mask[y[i]][x[i]] == 0:
            result_x.append(x[i])
            result_y.append(y[i])
            
    # return a list of x and y positions
    return result_x, result_y

def make_2d_array(x, y):
    arr = np.array([x, y]).T
    # change the dataype to float32
    arr = arr.astype(np.float64)
    return arr

def scale_data(data):
    """
    scale each columns of the 2d array to be between 0 and 1
    """
    data = np.array(data)
    for i in range(data.shape[1]):
        data[:,i] = (data[:,i] - data[:,i].min()) / (data[:,i].max() - data[:,i].min())
    return data

def normalize_data(data):
    """
    normalize each column of the data by standard deviation and mean
    """
    mean_1 , mean_2 = np.mean(data[:,0]), np.mean(data[:,1])
    std_1 , std_2 = np.std(data[:,0]), np.std(data[:,1])
    data[:,0] = (data[:,0] - mean_1)/std_1
    data[:,1] = (data[:,1] - mean_2)/std_2
    return data

def make_swirl_data(N = 2000, angle_start =(-1/2)*np.pi , angle_end =(5/2)*np.pi):
    """generates a helix"""
    
    theta = np.linspace(angle_start,angle_end,N) #np.sqrt(np.random.rand(N))*2*pi # 

    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(N,2)

    return x_a


def make_2d_array(x, y):
    arr = np.array([x, y]).T
    # change the dataype to float32
    arr = arr.astype(np.float64)
    return arr

def scale_data(data):
    """
    scale each columns of the 2d array to be between 0 and 1
    """
    data = np.array(data)
    for i in range(data.shape[1]):
        data[:,i] = (data[:,i] - data[:,i].min()) / (data[:,i].max() - data[:,i].min())
    return data

def normalize_data(data):
    """
    normalize each column of the data by standard deviation and mean
    """
    mean_1 , mean_2 = np.mean(data[:,0]), np.mean(data[:,1])
    std_1 , std_2 = np.std(data[:,0]), np.std(data[:,1])
    data[:,0] = (data[:,0] - mean_1)/std_1
    data[:,1] = (data[:,1] - mean_2)/std_2
    return data

def make_swirl_data(N = 2000, angle_start =(-1/2)*np.pi , angle_end =(5/2)*np.pi):
    """generates a helix"""
    
    theta = np.linspace(angle_start,angle_end,N) #np.sqrt(np.random.rand(N))*2*pi # 

    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(N,2)

    return x_a


def plot_all(prob_dist):
    nrows = 2
    ncols = 5
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 6))
    for i in range(nrows):
        for j in range(ncols):
            ax[i][j].scatter(prob_dist[i*ncols+j][:,0], prob_dist[i*ncols+j][:,1], s=1)
    
    plt.show()
  
def generate_data():
    """
    generates two classes of distributions p0 and p1
    p0: swirl data
    p1: 2d array data
    """
    p0_data_list, p1_data_list = [], []
    list_1 = np.random.randint(20,35,10)
    list_2 = np.random.randint(1700,2000,10)
    for i in range(10):
        p0_data_list.append(scale_data(make_swirl_data(N=list_2[i])))
        p1_data_list.append(scale_data(make_2d_array(*get_masked_data('S', intensity=list_1[i], rand=True))))

    return p0_data_list, p1_data_list

def create_images(prob_dist ,type):
    """
    store all the images of a corresponding distribution in a folder 
    type: 'p0' or 'p1'
    prob_dist: list of 2d arrays
    """
    for i in range(len(prob_dist)):
        plt.scatter(prob_dist[i][:,0], prob_dist[i][:,1], s=1)
        plt.axis('off')
        plt.savefig(f'{type}_images/{type}_image_{i}.png')
        plt.close()

def create_dataset():
    """
    create a dataset of images for both the distributions
    """
    p0_data_list, p1_data_list = generate_data()
    create_images(p0_data_list,'p0')
    create_images(p1_data_list,'p1')


class swirl_dataset(Dataset):
    pass
