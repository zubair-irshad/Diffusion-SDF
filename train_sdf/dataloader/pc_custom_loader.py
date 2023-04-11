#!/usr/bin/env python3
import os
import torch
import torch.utils.data

import pandas as pd
import random

class PCloaderCustom(torch.utils.data.Dataset):

    def __init__(
        self,
        data_source,
        pc_size=1024,
    ):
        self.folders = os.listdir(data_source)
        self.data_source = data_source
        self.pc_size = pc_size

    def __getitem__(self, idx):
        # train_idx = random.randint(0, len(self.folders) - 1)
        pcd_file_name = os.path.join(self.data_source, self.folders[idx], 'pcd.csv')
        pc = self.get_pcd(pcd_file_name, pc_size=self.pc_size)
        return pc, 'couch', self.folders[idx]

    def __len__(self):
        return len(self.folders)


    def get_pcd(self, f, pc_size =1024):
        f=pd.read_csv(f, sep=',',header=None).values
        pc = torch.from_numpy(f)

        pc_idx = torch.randperm(pc.shape[0])[:pc_size]
        pc = pc[pc_idx].double()
        return pc