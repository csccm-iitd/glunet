import numpy as np
import pandas as pd
import argparse
import torch
import h5py
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Download Data from here
# !git clone "https://github.com/cics-nd/cnn-surrogate.git"
# !bash "./cnn-surrogate/scripts/download_dataset.sh" 4225


class Data(Dataset):
    def __init__(self,t1,t2,transform=None):
       self.t1=t1
       self.t2=t2
       self.transform=transform

    def __len__(self):
        assert self.t1.shape[0]==self.t2.shape[0]
        return self.t1.shape[0]

    def __getitem__(self, idx):
        x=self.t1[idx]
        y=self.t2[idx]
        #sample = {'input':self.t1[idx], 'output':self.t2[idx]}
        if self.transform:
            x=self.transform(x)
        return {'input':x,'output':y}


def preprocess(test_path, train_path, data_points, train_batch_size, test_batch_size, file):
    f2 = h5py.File(test_path, 'r')
    test=torch.Tensor(f2['input'])
    test_out=torch.Tensor(f2['output'])
    
    f = h5py.File(train_path, 'r')
    t=torch.Tensor(f['input'])
    t_out=torch.Tensor(f['output'])
    t=t[:datapoints,:,:,:]
    t_out=t_out[:datapoints,:,:,:]

    transform=transforms.Compose([
        transforms.Normalize(mean=[0],
                         std=[1])
    ])

    d1=Data(t,t_out)
    d2=Data(test,test_out)
    train_dataloader = DataLoader(d1, batch_size=train_batch_size, shuffle=False)
    test_dataloader=DataLoader(d2,batch_size=test_batch_size,shuffle=False)

    return train_dataloader, test_dataloader
