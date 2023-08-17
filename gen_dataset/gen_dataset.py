import os
import glob
import numpy as np
import imageio.v3 as imageio

# Predifined peak value of each scene
foot_dict = {'beerfest_lightshow_01': 60.717923174513146, 'beerfest_lightshow_02': 478.1178264212101, 'beerfest_lightshow_02_reconstruction_update_2015': 482.3360310006649, 'beerfest_lightshow_03': 35.939507230122885, 'beerfest_lightshow_04': 192.90464248502158,
             'beerfest_lightshow_04_reconstruction_update_2015': 204.72225637469143, 'beerfest_lightshow_05': 104.28169161081314, 'beerfest_lightshow_06': 351.03759219292965, 'beerfest_lightshow_07': 220.76632392406464, 'bistro_01': 174.7414870104253, 'bistro_02': 223.58788572711708, 'bistro_03': 86.55792842191808, 'carousel_fireworks_01': 364.19813082267046, 'carousel_fireworks_02': 498.7218083817829, 'carousel_fireworks_03': 493.635499359459, 'carousel_fireworks_04': 141.84006071090698, 'carousel_fireworks_05': 433.41774162204786, 'carousel_fireworks_06': 119.5637951222333, 'carousel_fireworks_07': 495.4070549533792, 'carousel_fireworks_08': 432.31907334229237, 'carousel_fireworks_09': 100.95291816104542, 'cars_closeshot': 429.8834358989329, 'cars_fullshot': 383.78515438580405, 'cars_longshot': 455.5022449028201, 'fireplace_01': 470.3036923457066, 'fireplace_02': 432.8508772777632, 'fishing_closeshot': 499.80078125, 'fishing_longshot': 353.30032906658073, 'hdr_testimage': 265.8702720006307, 'poker_fullshot': 233.58810709635418, 'poker_travelling_slowmotion': 318.52470106801803, 'showgirl_01': 396.09499288342664, 'showgirl_02': 367.95990511888635, 'smith_hammering': 476.6622160231531, 'smith_welding': 496.81805453153356}

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def hdr2ldr(hdrs):
    ''' Convert set of HDR image to set of 14bit LDR images.

    Args:
        hdrs (4 H W 3): set of HDR images.

    Returns:
        ldrs (4 H W 3): set of 14bit LDR images. Range of [0,1]
    '''
    hdrs = hdrs * (pow(2, 14)-1)
    ldrs = np.clip(hdrs, 0, (pow(2, 14)-1))
    ldrs = np.around(ldrs-0.5)/(pow(2, 14)-1)
    ldrs = np.clip(ldrs,0,1)
    return ldrs

def ldrs2raw(ldrs):
    ''' Convert LDR image set to raw Bayer image (linear).

    Args:
        ldrs (4 H W 3): set of 14bit LDR images (linear).

    Returns:
        raw_bayer (H W): subsampled raw Bayer image (linear).
        R0 R1 G0 G1
        R1 R2 G1 G2
        G0 G1 B0 B1
        G1 G2 B1 B2
    '''
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

dataset_name = 'froehlich'
frame_skip = 30
MAXVALUE = 100000
test_scene = ['cars_fullshot','poker_fullshot','showgirl_02']
method_dir = 'C:/Users/VCLAB/deblur_dataset/source/HDR_Camera_Footage/'

for data_dir in glob.glob('{}*\\'.format(method_dir)):
    scene_name = data_dir.split('\\')[-2]
    dataset_dir = 'C:/Users/VCLAB/deblur_dataset/source/HDR_Camera_Footage/{}/'.format(scene_name)

    for img_name_dir in glob.glob('{}*.exr'.format(dataset_dir)):
        frame_num = int(img_name_dir.split('\\')[-1].split('_')[-1][:-4]) # read frame number
        
        # Select frames
        if frame_num%frame_skip != 0:
            continue
            
        # Image name w/ frame number
        img_name = img_name_dir.split('\\')[-1][:-4]
        # Image name w/o frame number
        short_name = img_name_dir.split('\\')[-1][:-11]
        
        # Ground-truth HDR frame
        target_frame = imageio.imread('{}{}.exr'.format(dataset_dir,img_name))
        gt_frame = target_frame.copy()
        img_h,img_w,_ = target_frame.shape
        cur_frame_num = frame_num
        exp0 = np.clip(target_frame.copy(),0,MAXVALUE)
        
        # Get pre-calculated peak value
        target_frame_peak = foot_dict[scene_name]
        
        # GT dataset: [0,1] range
        target_frame = target_frame/target_frame_peak
        target_frame = np.clip(target_frame,0,1)
        
        # Save GT array
        if scene_name in test_scene:
            np.save('C:/Users/VCLAB/deblur_dataset/gt_hdr_test/{}.npy'.format(img_name),target_frame)
        else:
            np.save('C:/Users/VCLAB/deblur_dataset/gt_hdr_train/{}.npy'.format(img_name),target_frame)
        
        # Generate HDR with various exposure time
        try:
            exp1 = exp0.copy()
            for i in range(3):
                cur_frame_num += 1
                exp1 += np.clip(imageio.imread('%s%s_%06d.exr'%(dataset_dir,short_name,(cur_frame_num))),0,MAXVALUE)
            
            exp2 = exp1.copy()
            for i in range(12):
                cur_frame_num += 1
                exp2 += np.clip(imageio.imread('%s%s_%06d.exr'%(dataset_dir,short_name,(cur_frame_num))),0,MAXVALUE)

        except FileNotFoundError:
            break
        
        hdrs = np.clip(np.stack((exp0,exp1,exp1,exp2),axis=0)/target_frame_peak,0,1) # shape=(4,H,W,3)
        ldrs = hdr2ldr(hdrs) # shape=(4,H,W,3)
        output = ldrs2raw(ldrs) # shape=(H,W)
        output = np.clip(output,0,1)

        if scene_name in test_scene:
            np.save('C:/Users/VCLAB/deblur_dataset/input_test/{}.npy'.format(img_name),output)
        else:
            np.save('C:/Users/VCLAB/deblur_dataset/input_train/{}.npy'.format(img_name),output)

    