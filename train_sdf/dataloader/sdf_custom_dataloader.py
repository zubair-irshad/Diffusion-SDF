#!/usr/bin/env python3
import os
import torch
import torch.utils.data

import pandas as pd 
import numpy as np

class SdfLoaderCustom(torch.utils.data.Dataset):

    def __init__(
        self,
        data_source, # path to points sampled around surface
        samples_per_mesh=16000,
        pc_size=1024,
        modulation_path=None # used for third stage of training; needs to be set in config file when some modulation training had been filtered
    ):

        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size
        self.folders = os.listdir(data_source)
        self.data_source = data_source

    def get_pcd(self, f, pc_size =1024):
        f=pd.read_csv(f, sep=',',header=None).values
        pc = torch.from_numpy(f)

        pc_idx = torch.randperm(pc.shape[0])[:pc_size]
        pc = pc[pc_idx]

        return pc

    def labeled_sampling(self, f, subsample):

        f=pd.read_csv(f, sep=',',header=None).values
        f = torch.from_numpy(f)
        half = int(subsample / 2) 
        neg_tensor = f[f[:,-1]<0]
        pos_tensor = f[f[:,-1]>0]

        if pos_tensor.shape[0] < half:
            pos_idx = torch.randint(0, pos_tensor.shape[0], (half,))
        else:
            pos_idx = torch.randperm(pos_tensor.shape[0])[:half]

        if neg_tensor.shape[0] < half:
            if neg_tensor.shape[0]==0:
                neg_idx = torch.randperm(pos_tensor.shape[0])[:half] # no neg indices, then just fill with positive samples
            else:
                neg_idx = torch.randint(0, neg_tensor.shape[0], (half,))
        else:
            neg_idx = torch.randperm(neg_tensor.shape[0])[:half]

        pos_sample = pos_tensor[pos_idx]

        if neg_tensor.shape[0]==0:
            neg_sample = pos_tensor[neg_idx]
        else:
            neg_sample = neg_tensor[neg_idx]

        samples = torch.cat([pos_sample, neg_sample], 0)

        return samples[:,:3].float().squeeze(), samples[:, 3].float().squeeze() # pc, xyz, sdv


    def __getitem__(self, idx): 

        near_surface_count = int(self.samples_per_mesh*0.7)

        surface_file_name = os.path.join(self.data_source, self.folders[idx], 'surface.csv')
        grid_file_name = os.path.join(self.data_source, self.folders[idx], 'uniform.csv')
        pcd_file_name = os.path.join(self.data_source, self.folders[idx], 'pcd.csv')

        sdf_xyz, sdf_gt =  self.labeled_sampling(surface_file_name, near_surface_count)
        
        grid_count = self.samples_per_mesh - near_surface_count
        grid_xyz, grid_gt = self.labeled_sampling(grid_file_name, grid_count)
        sdf_xyz = torch.cat((sdf_xyz, grid_xyz))
        sdf_gt = torch.cat((sdf_gt, grid_gt))

        pc = self.get_pcd(pcd_file_name, pc_size=self.pc_size)
    
        data_dict = {
                    "xyz":sdf_xyz.float().squeeze(),
                    "gt_sdf":sdf_gt.float().squeeze(), 
                    "point_cloud":pc.float().squeeze(),
                    }

        return data_dict

    def __len__(self):
        return len(self.folders)