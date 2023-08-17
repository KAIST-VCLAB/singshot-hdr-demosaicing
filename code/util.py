import os
import pdb
import sys
import torch

# For debugging in mp
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used from a forked multiprocessing child.
    
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def load_ckpt(net, ckpt_path):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=net.device)
        net.load_state_dict(ckpt)
        print('# Checkpoint loaded!')
    else:
        raise ValueError('No checkpoint exists')

# Remove module from keys in state_dict (e.g. module.conv_x1.weight -> conv_x1.weight)
def load_ckpt_ddp(net, ckpt_path, device):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_weight = ckpt['model_state_dict']
        if list(ckpt_weight.keys())[0].split('.')[0] == 'module':
            ckpt_ddp = {'.'.join(k.split('.')[1:]): v for k,v in ckpt_weight.items()}
            net.load_state_dict(ckpt_ddp)
        else:
            net.load_state_dict(ckpt)
        print('Checkpoint loaded!')
    else:
        raise ValueError('No checkpoint exists')