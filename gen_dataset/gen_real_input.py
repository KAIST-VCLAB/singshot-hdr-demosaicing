import os
import numpy as np
import glob
import rawpy
from tqdm import tqdm 
from joblib import Parallel, delayed

real_dataset_dir = 'C:\\Users\\VCLAB\\cvpr\\burst\\burst_data\\1110_1'
save_dir = 'C:\\Users\\VCLAB\\deblur_dataset\\result\\burst_input'

def make_dir(dir_path):
    new_dir = os.path.normpath(dir_path)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir

def ldrs2raw(ldrs):
    _,h,w,_ = ldrs.shape
    ldrs = ldrs[:,:h//4*4,:w//4*4,:]
    raw_bayer = np.zeros((h//4*4,w//4*4))

    for idx in range(4):
        offset_h = idx//2
        offset_w = idx%2
        raw_bayer[offset_h::4,offset_w::4] = ldrs[idx,offset_h::4,offset_w::4,0]
        raw_bayer[2+offset_h::4,offset_w::4] = ldrs[idx,2+offset_h::4,offset_w::4,1]
        raw_bayer[offset_h::4,2+offset_w::4] = ldrs[idx,offset_h::4,2+offset_w::4,1]
        raw_bayer[2+offset_h::4,2+offset_w::4] = ldrs[idx,2+offset_h::4,2+offset_w::4,2]
    return raw_bayer

def generate_raw(file_):
    frame_target = [1,4,4,16]
    max_frame = max(frame_target)
    filename = os.path.split(file_)[-1]

    # Get single frame with shortest exposure time
    demosaiced_img = rawpy.imread(os.path.join(new_dir,filename)).postprocess(
        gamma = (1,1),
        half_size = True,
        four_color_rgb = True,
        fbdd_noise_reduction = rawpy.FBDDNoiseReductionMode(1),
        median_filter_passes = 2,
        use_camera_wb = True,
        use_auto_wb = False,
        noise_thr = 100,
        no_auto_bright=True,
        demosaic_algorithm = rawpy.DemosaicAlgorithm(3),
        output_bps=16,
        output_color=rawpy.ColorSpace(1))
    demosaiced_img = np.float32(demosaiced_img/65535.0)
    h,w,c = demosaiced_img.shape
    frame_group = np.zeros((max_frame,h,w,c),np.float32)
    frame_group[0,:,:,:] = demosaiced_img

    ldr_group = np.zeros((4,h,w,c),np.float32)
    ldr_group[0,:,:,:] = demosaiced_img
    
    origin_filename = os.path.splitext(os.path.split(file_)[-1])[0]
    origin_filenumber = int(origin_filename[-4:])
    
    # Combine multiple frames
    for i in range(1,max_frame):
        temp_filename = 'GO5A' + str(origin_filenumber + i).zfill(4) + '.CR2'
        temp_demosaiced_img = rawpy.imread(os.path.join(new_dir,temp_filename)).postprocess(
            gamma = (1,1),
            half_size = True,
            four_color_rgb = True,
            fbdd_noise_reduction = rawpy.FBDDNoiseReductionMode(1),
            median_filter_passes = 2,
            use_camera_wb = True,
            use_auto_wb = False,
            noise_thr = 100,
            no_auto_bright=True,
            demosaic_algorithm = rawpy.DemosaicAlgorithm(3),
            output_bps=16,
            output_color=rawpy.ColorSpace(1))
        temp_demosaiced_img = np.float32(temp_demosaiced_img/65535.0)
        frame_group[i,:,:,:] = temp_demosaiced_img
    
    for i,frame in enumerate(frame_target):
        sum_frame = np.float32(np.zeros((h,w,c)))
        for idx in range(frame):
            sum_frame += frame_group[idx,:,:,:]
        ldr_group[i,:,:,:] = sum_frame

    result = ldrs2raw(ldr_group)
    result = np.clip(result,0,1)
    np.save(f'{save_dir}\\{origin_filename}.npy',result)

num_cores = 10
for i in range(10):
    new_dir = f'{real_dataset_dir}\\{i}'
    new_view_dir = f'{real_dataset_dir}\\{i}-view\\'
    
    make_dir(new_view_dir)
    files = glob.glob(os.path.join(new_dir, '*.CR2'))[:-15]
    Parallel(n_jobs=num_cores)(delayed(generate_raw)(file_) for file_ in tqdm(files))
