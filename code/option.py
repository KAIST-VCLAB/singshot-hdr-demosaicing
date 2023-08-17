import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# Optimization specifications
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')

# Train specifications
parser.add_argument('--num_epoch', type=int, default=200, help='number of epochs to train')

parser.add_argument('--input_size', type=int, default=256, help='input size')    

parser.add_argument('--batch_size', type=int, default=8, help='batch size')                             

parser.add_argument('--is_noise', type=bool, default=True, help='noise on/off')   

parser.add_argument('--noise_value', type=float, default=1.0, help='noise scale')                               

parser.add_argument('--is_resume', type=bool, default=False, help='resume on/off')    

parser.add_argument('--is_train', type=str2bool, default=True, help='train on/off')     

# Ablation
parser.add_argument('--demosaicing', type=str2bool, default=True, help='demosaicing on/off')         

parser.add_argument('--multiscale', type=str2bool, default=True, help='multiscale on/off')       

# Burst
parser.add_argument('--burst', action='store_true', default=False)

# Note       
parser.add_argument('--result_dir', type=str, default='base_dir', help='dir for model')                  

args = parser.parse_args()