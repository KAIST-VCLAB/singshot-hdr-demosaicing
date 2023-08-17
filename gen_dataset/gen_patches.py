import os
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm

##### From Restormer: generate_patches_sidd.py #####
patch_size = 256
overlap = 64
padding = 12

base_dir = 'C:\\Users\\VCLAB\\deblur_dataset\\'
gt_dir = os.path.join(base_dir, 'gt_hdr_train')
bayer_dir = os.path.join(base_dir, 'input_train')
files = natsorted(glob(os.path.join(bayer_dir, '*.npy')))

def save_files(file_):
    filename = os.path.split(file_)[-1]
    
    bayer_img = np.load(os.path.join(bayer_dir,filename))
    gt_img = np.load(os.path.join(gt_dir,filename))
    filename = os.path.splitext(os.path.split(file_)[-1])[0]
    num_patch = 0
    w, h, _ = gt_img.shape
    w, h = bayer_img.shape
    
    w1 = list(np.arange(padding, w-patch_size-padding, patch_size-overlap, dtype=np.int32))
    h1 = list(np.arange(padding, h-patch_size-padding, patch_size-overlap, dtype=np.int32))
    w1.append(w-patch_size-padding)
    h1.append(h-patch_size-padding)
    for i in w1:
        for j in h1:
            num_patch += 1
                
            lr_patch = bayer_img[i:i+patch_size, j:j+patch_size]
            hr_patch = gt_img[i:i+patch_size, j:j+patch_size, :]
            
            lr_savename = os.path.join(base_dir, 'input_train_cropped', filename + '-' + str(num_patch) + '.npy')
            hr_savename = os.path.join(base_dir, 'gt_hdr_train_cropped', filename + '-' + str(num_patch) + '.npy')
            
            np.save(lr_savename, lr_patch) 
            np.save(hr_savename, hr_patch)

from joblib import Parallel, delayed
num_cores = 10
Parallel(n_jobs=num_cores)(delayed(save_files)(file_) for file_ in tqdm(files))