import argparse
from model import UNET
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from torch.autograd import Variable
import dill
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence
from scipy.special import logit as slogit
from typing import Callable, Optional, Sequence, Union
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from torchvision import transforms
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch.nn.init as init
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def train(train_dataloader, test_dataloader, lr, weight_decay, num_epochs, model_save_path, use_gpu, f):
    model= UNET(in_channels=1, out_channels=3)
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write('Number of parameters: %d' % num_params)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    loss=nn.MSELoss()
    model.train()
    train_loss_avg = []
    loss_list=[]

    for epoch in range(num_epochs):
      for index,images in enumerate(train_dataloader):
        image_batch=images["input"]
        image_batch_output=images["output"]
        image_batch = image_batch.to(device)
        image_batch_recon= model(image_batch)
        loss_val = loss(image_batch_recon,image_batch_output)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        f.write(loss_val.data)
        loss_list.append(loss_val.data)
      with torch.no_grad():
        for index,images in enumerate(test_dataloader):
          image_batch=images["input"]
          image_batch_output=images["output"]
          image_batch = image_batch.to(device)
          image_batch_recon= model(image_batch)
          f.write("Test Loss ----------> ",loss(image_batch_recon,image_batch_output).data)
          f.write("Test R2 Pressure   ----------> ",r2_score(image_batch_output[:,0,:,:].flatten().detach().numpy(),image_batch_recon[:,0,:,:].flatten().detach().numpy()))
          f.write("Test R2 Vel X      ----------> ",r2_score(image_batch_output[:,1,:,:].flatten().detach().numpy(),image_batch_recon[:,1,:,:].flatten().detach().numpy()))
          f.write("Test R2 Vel Y      ----------> ",r2_score(image_batch_output[:,2,:,:].flatten().detach().numpy(),image_batch_recon[:,2,:,:].flatten().detach().numpy()))
          f.write("Test R2            ----------> ",r2_score(image_batch_output.flatten().detach().numpy(),image_batch_recon.flatten().detach().numpy()))
      f.write("Epoch Number ",epoch," done.")
    torch.save(model, model_save_path)
    return loss_list