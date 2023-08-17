import shutil
import glob
import imageio.v3 as imageio
import random

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import dataset
import network
from util import *
from option import args
from metric import *

def main():
    print('Start demo')
    rank = 0
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = network.make_model(args,rank).cuda(rank)
    torch.cuda.set_device(rank)

    # Load pre-trained model
    load_ckpt_ddp(net=net, ckpt_path='models/best_psnr_mu.pt',device=dev)

    # Generator
    test_bayers = ['cars_fullshot_000690']
    test_gts = ['cars_fullshot_000690_gt']
    test_dataset_dir = 'demo/data'
    test_gt_dir = 'demo/data'
    test_dataset = dataset.Dataset(test_bayers, test_gts, test_dataset_dir, 
                                   test_gt_dir, args, is_test=True, is_burst=args.burst)
    testing_generator = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False,
                                                    num_workers=0, pin_memory=True)
    
    with torch.no_grad():
        net.eval()
        for data_group in testing_generator:
            data_cuda = [x.float().cuda(rank, non_blocking=True) for x in data_group[:2]]
            input_bayer, gt_hdr = data_cuda # input_bayer: (b h w 2)
            pred_hdr = net(input_bayer).permute(0,2,3,1) # pred_hdr: (b h w c)
            input_name = data_group[2][0]

            cur_psnr = batch_psnr(pred_hdr, gt_hdr, 1.0)
            mu_pred_hdr = range_compressor_cuda(pred_hdr)
            mu_gt_hdr = range_compressor_cuda(gt_hdr)
            cur_psnr_mu = batch_psnr(mu_pred_hdr, mu_gt_hdr,1.0)

            pred_hdr = np.clip(np.squeeze(pred_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)
            gt_hdr = np.clip(np.squeeze(gt_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)
            mu_pred_hdr = np.clip(np.squeeze(mu_pred_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)
            mu_gt_hdr = np.clip(np.squeeze(mu_gt_hdr.detach().cpu().numpy(),0) * 255.0, 0., 255.)

            cur_ssim_mu = calculate_ssim(mu_pred_hdr, mu_gt_hdr)
            
            imageio.imwrite(f'demo/results/{input_name}_pred.png',np.uint8(mu_pred_hdr))
            imageio.imwrite(f'demo/results/{input_name}_gt.png',np.uint8(mu_gt_hdr))

            print('Result image Saved')
            print(f"PSNR: {cur_psnr}, PSNR-mu: {cur_psnr_mu}, SSIM-mu {cur_ssim_mu}")

if __name__ == '__main__':
    main()