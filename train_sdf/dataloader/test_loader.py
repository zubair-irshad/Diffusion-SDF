#!/usr/bin/env python3
import numpy as np
import time
import logging
import os
import random
import torch
import torch.utils.data
import sys
sys.path.append('/home/zubairirshad/Diffusion-SDF')
from train_sdf.dataloader import base
import open3d as o3d
import pandas as pd
import csv
import open3d as o3d
import numpy as np
points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
])- 0.5
lines = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
class TestAcronymDataset(base.Dataset):
    def __init__(
        self,
        data_source,
        split_file, # json filepath which contains train/test classes and meshes
        pc_size,
        gt_filename= "sdf_data.csv",
        samples_per_mesh = 16000
    ):
        # super().__init__(data_source, split_file, pc_size, gt_filename, samples_per_mesh)
        # self.gt_files = self.get_instance_filenames(data_source, split_file, gt_filename)
        self.pc_size = pc_size
        print("pc_size", pc_size)
        self.samples_per_mesh = samples_per_mesh
        self.grid_source = None
    def __getitem__(self, idx):
        np.random.seed()
        sdf_file = '/home/zubairirshad/Diffusion-SDF/train_sdf/data/grid_data/acronym/Couch/37cfcafe606611d81246538126da07a8/grid_gt.csv'
        gt_file = '/home/zubairirshad/Diffusion-SDF/train_sdf/data/acronym/Couch/37cfcafe606611d81246538126da07a8/sdf_data.csv'
        near_surface_count = int(self.samples_per_mesh*0.7)
        #pc, sdf_xyz, sdf_gt =  self.labeled_sampling(self.gt_files[idx], self.subsample, self.pc_size)
        #pc, sdf_xyz, sdf_gt  = self.labeled_sampling(gt_file, near_surface_count, self.pc_size)
        pc, sdf_xyz, sdf_gt =  self.labeled_sampling(gt_file, near_surface_count, pc_size = self.pc_size)
        print("sdf_xyz", pc.shape, sdf_xyz.shape, sdf_gt.shape)
        grid_count = self.samples_per_mesh - near_surface_count
        _, grid_xyz, grid_gt = self.labeled_sampling(sdf_file, grid_count, pc_size=1024)
        print("grid_xyz", grid_xyz.shape, grid_gt.shape)
        # each getitem is one batch so no batch dimension, only N, 3 for xyz or N for gt
        # for 16000 points per batch, near surface is 11200, grid is 4800
        #print("shapes: ", pc.shape,  sdf_xyz.shape, sdf_gt.shape, grid_xyz.shape, grid_gt.shape)
        sdf_xyz = torch.cat((sdf_xyz, grid_xyz))
        sdf_gt = torch.cat((sdf_gt, grid_gt))
        # mesh_name = gt_file.split("/")[-3:-1] # class and mesh
        # mesh_name = os.path.join(mesh_name[0],mesh_name[1])
        data_dict = {"point_cloud":pc,
                    "xyz":sdf_xyz,
                    "gt_sdf":sdf_gt,
                    "indices":idx
                    }
        return data_dict
    def __len__(self):
        # return len(self.gt_files)
        return 1
if __name__ == "__main__":
    test_data = TestAcronymDataset(data_source="data",
                                   split_file="data/splits/couch_single.json",
                                   pc_size=1024,
                                   gt_filename="sdf_data.csv",
                                )
    train_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=1, num_workers=1,
            drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True
        )
    for i, data in enumerate(train_dataloader):
        for k,v in data.items():
            if torch.is_tensor(v):
                print(k, v.shape)
        pc = data['point_cloud'].squeeze().numpy()
        sdf_xyz = data['xyz'].squeeze().numpy()
        sdf_gt = data['gt_sdf'].squeeze().numpy()
        colors = np.zeros(sdf_xyz.shape)
        colors[sdf_gt < 0, 2] = 1
        colors[sdf_gt > 0, 0] = 1
        print("sdf_xyz", sdf_xyz.shape, sdf_gt.shape)
        o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc)), line_set])
        pcd_gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sdf_xyz))
        pcd_gt.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd_gt, line_set])



    
