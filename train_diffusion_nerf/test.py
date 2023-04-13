#!/usr/bin/env python3

import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

import os
import json, csv
import time
from tqdm.auto import tqdm
from einops import rearrange, reduce
import numpy as np
import trimesh
import warnings

# add paths in model/__init__.py for new models
from models import * 
from utils import mesh
from utils.reconstruct import *
from diff_utils.helpers import * 
#from metrics.evaluation_metrics import *#compute_all_metrics
#from metrics import evaluation_metrics

# from dataloader.pc_loader import PCloader
from dataloader.pc_custom_loader import PCloaderCustom as PCloader

           
@torch.no_grad()
def test_generation():

    # load model 
    # if args.resume == 'finetune': # after second stage of training 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #=============================================================================="
        #=============================================================================="
        # load NeRF model here if we want to do combine NeRF based rendering with diffusion based generation
        #=============================================================================="
        #=============================================================================="
       
        # model = CombinedModel.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False) 
        # #model = model.load_from_checkpoint(specs["diffusion_ckpt_path"], specs=specs, strict=False)

        ckpt = torch.load(specs["diffusion_ckpt_path"])
        model.diffusion_model.load_state_dict(ckpt['model_state_dict'])
        model = model.cuda().eval()
    # else:
    #     # ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    #     # resume = os.path.join(args.exp_dir, ckpt)
    #     # model = CombinedModel.load_from_checkpoint(resume, specs=specs).cuda().eval()

    conditional = specs["diffusion_model_specs"]["cond"] 

    # if not conditional:
    
    #currently only saving samples to a file so we can render them using NeRF script in other repos.
    #Currently only doing unconditional generation

    samples = model.diffusion_model.generate_unconditional(args.num_samples)

    print("samples shape: ", samples.shape)


        
        #=============================================================================="
        #=============================================================================="
        # Render NeRF model here based on samples obtained
        #=============================================================================="
        #=============================================================================="
        
        # plane_features = model.vae_model.decode(samples)
        # for i in range(plane_features.shape[0]):
        #     plane_feature = plane_features[i].unsqueeze(0)
        #     mesh.create_mesh(model.sdf_model, plane_feature, recon_dir+"/{}_recon".format(i), N=128, max_batch=2**21, from_plane_features=True)

    # else:
    #     # load dataset, dataloader, model checkpoint
    #     # test_split = json.load(open(specs["TestSplit"]))
    #     test_dataset = PCloader(specs["DataSource"], pc_size=specs.get("PCsize",1024))
    #     # test_dataset = PCloader(specs["DataSource"], test_split, pc_size=specs.get("PCsize",1024), return_filename=True)
    #     test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)

    #     with tqdm(test_dataloader) as pbar:
    #         for idx, data in enumerate(pbar):
    #             pbar.set_description("Files generated: {}/{}".format(idx, len(test_dataloader)))

    #             point_cloud, cls_name, mesh_name = data # filename = path to the csv file of sdf data
    #             # point_cloud, filename = data # filename = path to the csv file of sdf data
    #             filename = filename[0] # filename is a tuple

    #             # cls_name = filename.split("/")[-3]
    #             # mesh_name = filename.split("/")[-2]
    #             outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
    #             os.makedirs(outdir, exist_ok=True)

    #             # filter, set threshold manually after a few visualizations
    #             if args.filter:
    #                 threshold = 0.08
    #                 tmp_lst = []
    #                 count = 0
    #                 while len(tmp_lst)<args.num_samples:
    #                     count+=1
    #                     samples, perturbed_pc = model.diffusion_model.generate_from_pc(point_cloud.cuda(), batch=args.num_samples, save_pc=outdir, return_pc=True) # batch should be set to max number GPU can hold
    #                     plane_features = model.vae_model.decode(samples)
    #                     # predicting the sdf values of the point cloud
    #                     perturbed_pc_pred = model.sdf_model.forward_with_plane_features(plane_features, perturbed_pc.repeat(args.num_samples, 1, 1))
    #                     consistency = F.l1_loss(perturbed_pc_pred, torch.zeros_like(perturbed_pc_pred), reduction='none')
    #                     loss = reduce(consistency, 'b ... -> b', 'mean', b = consistency.shape[0]) # one value per generated sample 
    #                     #print("consistency shape: ", consistency.shape, loss.shape, consistency[0].mean(), consistency[1].mean(), loss) # cons: [B,N]; loss: [B]
    #                     thresh_idx = loss<=threshold
    #                     tmp_lst.extend(plane_features[thresh_idx])

    #                     if count > 5: # repeat this filtering process as needed 
    #                         break
    #                 # skip the point cloud if cannot produce consistent samples or 
    #                 # just use the samples that are produced if comparing to other methods
    #                 if len(tmp_lst)<1: 
    #                     continue
    #                 plane_features = tmp_lst[0:min(10,len(tmp_lst))]

    #             else:
    #                 # for each point cloud, the partial pc and its conditional generations are all saved in the same directory 
    #                 samples, perturbed_pc = model.diffusion_model.generate_from_pc(point_cloud.cuda(), batch=args.num_samples, save_pc=outdir, return_pc=True)
    #                 plane_features = model.vae_model.decode(samples)
                
    #             for i in range(len(plane_features)):
    #                 plane_feature = plane_features[i].unsqueeze(0)
    #                 mesh.create_mesh(model.sdf_model, plane_feature, outdir+"/{}_recon".format(i), N=128, max_batch=2**21, from_plane_features=True)
            


    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r", default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    arg_parser.add_argument("--num_samples", "-n", default=50, type=int, help='number of samples to generate and reconstruct')

    arg_parser.add_argument("--filter", default=False, help='whether to filter when sampling conditionally')

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"])

    output_dir = '/experiments/zubair/shapenet/diffusion_sdf'

    # recon_dir = os.path.join(args.exp_dir, "recon")
    recon_dir = os.path.join(output_dir, "recon_unconditional_stage2")
    os.makedirs(recon_dir, exist_ok=True)
    
    test_generation()
