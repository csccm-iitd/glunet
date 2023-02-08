import argparse
from model import UNET
from train import train
from preprocess import preprocess
from eval import loss_plot, plot_from_model, draw_pdf
import json
import torch
import h5py
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

class Args:
    def __init__(self, params):
        for key in params.keys():
            setattr(self, key, params[key])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help = "configuration file path", type = str, required = True)
    config_args = parser.parse_args()
    config_data = json.loads(open(config_args.config_file, "r", encoding = "utf-8").read().strip())
    args = Args(config_data)

    f = open(args.log_output_file, "w+")

    for key in config_data.keys():
      f.write(key)
      f.write(str(config_data[key]))
    
    if args.train:
      train_dataloader, test_dataloader = preprocess(args.train_path, args.test_path, args.data_points, args.train_batch_size, args.test_batch_size, f)
      loss_list = train(train_dataloader, test_dataloader, args.lr, args.weight_decay, args.num_epochs, args.model_save_path, args.use_gpu, f)
      loss_plot(loss_list, args.model_load_path, f)
    plot_from_model(args.model_load_path, args.kle, args.datapoints, f)
    draw_pdf(args.model_load_path, args.kle, args.datapoints, f)
    f.close()