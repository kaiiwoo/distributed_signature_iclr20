import torch
import numpy as np
from datetime import datetime

def rand_init(*args, requires_grad=False, device='cuda'):
    return torch.rand(*args, dtype=torch.float, requires_grad=requires_grad, device=device)

# convert np.ndarray value to tensor value (in dictionary)
def convert_tensor(input_dict, device='cuda'):
    for v in input_dict.values():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if torch.is_tensor(v):
            v = v.to(device)
    return input_dict
    
    
def with_time(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    return str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '_' + s


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

