import random
import torch
import numpy as np
from util import ForkedPdb

class Dataset(torch.utils.data.Dataset):
    def __init__(self, bayer_ids, gt_ids, bayer_path, gt_path, args, is_test=False, is_burst = False):
        self.bayer_ids = bayer_ids
        self.gt_ids = gt_ids
        self.bayer_path = bayer_path
        self.gt_path = gt_path
        self.input_size = args.input_size
        self.noise_value = args.noise_value
        self.is_train = args.is_train
        self.is_test = is_test
        self.is_burst = is_burst
        
        # Normalize with exposure time
        exp = np.zeros((args.input_size,args.input_size))
        exp[::2,::2] = 1.0
        exp[::2,1::2] = 4.0
        exp[1::2,::2] = 4.0
        exp[1::2,1::2] = 16.0
        self.exp_level_arr = exp

    def __len__(self):
        return len(self.bayer_ids)

    def __getitem__(self, index):
        bayer_id = self.bayer_ids[index]
        gt_id = self.gt_ids[index]
        if self.is_burst:
            input_bayer = self._load_burst_input(bayer_id)
        else:
            input_bayer, gt_hdr = self._load_file(bayer_id, gt_id)
            if self.is_test:
                input_bayer = input_bayer[12:-12,16:-16]
                gt_hdr = gt_hdr[12:-12,16:-16,:]
            else:
                input_bayer, gt_hdr = self._augmentation(input_bayer, gt_hdr)

            if self.is_train:
                input_bayer = self._add_train_noise(input_bayer)
            else:
                input_bayer = self._add_noise(input_bayer)
                    
            input_bayer = np.clip(input_bayer,0,1)
            gt_hdr = np.clip(gt_hdr,0,1)
        
        h,w = input_bayer.shape
        # Normalize with exposure time
        if self.is_test:
            exp = np.zeros((h,w))
            exp[::2,::2] = 1.0
            exp[::2,1::2] = 4.0
            exp[1::2,::2] = 4.0
            exp[1::2,1::2] = 16.0
            self.exp_level_arr = exp

        input_bayer_gamma = np.expand_dims((input_bayer) / (self.exp_level_arr + 1e-8), axis=-1)
        input_bayer = np.expand_dims(input_bayer,axis=-1)
        input_bayer = np.concatenate((input_bayer,input_bayer_gamma),-1)
        assert(input_bayer.shape==(h,w,2))

        if self.is_burst:
            gt_hdr = input_bayer

        return input_bayer, gt_hdr, bayer_id
    
    def _load_file(self, bayer_id, gt_id):
        bayer_name = '{}/{}.npy'.format(self.bayer_path,bayer_id)
        gt_name = '{}/{}.npy'.format(self.gt_path,gt_id)
        bayer = np.load(bayer_name,allow_pickle=True)
        gt = np.load(gt_name,allow_pickle=True)
        return bayer, gt
    
    def _load_burst_input(self, bayer_id):
        bayer_name = '{}/{}.npy'.format(self.bayer_path,bayer_id)
        bayer = np.load(bayer_name,allow_pickle=True)
        return bayer
    
    def _augmentation(self, bayer, gt_hdr, hflip=True, rot=True):
        ''' Add random augmentation: Flip, Rotation.
        
        ''' 
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5 

        if hflip: bayer = bayer[:, ::-1].copy()
        if vflip: bayer = bayer[::-1, :].copy()
        if rot90: bayer = np.transpose(bayer).copy()

        if hflip: gt_hdr = gt_hdr[:, ::-1, :].copy()
        if vflip: gt_hdr = gt_hdr[::-1, :, :].copy()
        if rot90: gt_hdr = np.transpose(gt_hdr, (1, 0, 2)).copy()
        
        return bayer, gt_hdr

    def _add_train_noise(self, patch):
        ''' Add noise to raw bayer iamge for training.
    
        '''
        patch_h, patch_w = patch.shape
        noises = np.random.normal(0.0, scale=random.uniform(self.noise_value*0.9,self.noise_value*1.1), size=(patch_h, patch_w))
        patch += np.sqrt((patch*16383.0 + pow(34.9,2))/pow(5.04,2))/16383.0 * noises
        return patch

    
    def _add_noise(self, patch):
        ''' Add noise to raw bayer iamge for testing.
        
        '''
        noise_h, noise_w = patch.shape
        noises = np.random.normal(0.0, scale=self.noise_value, size=(noise_h, noise_w))
        patch += np.sqrt((patch*16383.0 + pow(34.9,2))/pow(5.04,2))/16383.0 * noises
        return patch
    
