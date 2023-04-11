#!/usr/bin/env python3

import time 
import logging
import os
import random
import torch
import torch.utils.data
from diff_utils.helpers import * 

import pandas as pd 
import numpy as np
import csv, json

from tqdm import tqdm

class ModulationLoaderCustom(torch.utils.data.Dataset):
    def __init__(self, data_path, pc_path=None, split_file=None, pc_size=None):
        super().__init__()

        self.folders = os.listdir(data_path)
        self.conditional = pc_path is not None 
        self.pc_size = pc_size

        self.data_source = data_path
        self.pc_source = pc_path
        if self.conditional:
            self.modulations, pc_paths = self.load_modulations(data_path, pc_path)
        else:
            self.modulations = self.unconditional_load_modulations(data_path)
        
    def get_pcd(self, f, pc_size =1024):
        f=pd.read_csv(f, sep=',',header=None).values
        pc = torch.from_numpy(f)

        pc_idx = torch.randperm(pc.shape[0])[:pc_size]
        pc = pc[pc_idx]
        return pc
    
    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        
        modulation_filename = os.path.join(self.data_source, self.folders[index], 'latent.txt')
        latent = torch.from_numpy(np.loadtxt(modulation_filename)).float()

        if self.conditional:
            pcd_file_name = os.path.join(self.pc_source, self.folders[index], 'pcd.csv')
            pc = self.get_pcd(pcd_file_name, pc_size=self.pc_size)
        else:
            pc = False
        return {
            "point_cloud" : pc,
            "latent" : latent         
        }