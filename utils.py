import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


def debug_1dim(net):
    lst_x, lst_y = [], []
    for i in range(-5,5,0.1):
        x = torch.tensor([i])
        y = net(x)
        lst_x.append(x)
        lst_y.append(y)
    plt.scatter(lst_x, lst_y)
    plt.show()
    
    
def init_uniform(component):
    if isinstance(component, list):
        for c in component:
            init_uniform(c)
    elif isinstance(component, torch.Tensor):
        nn.init.uniform_(component, a=-0.00001, b=0.00001)
    else:
        for param in component.parameters():
            nn.init.uniform_(param, a=-0.00001, b=0.00001)

def get_best_gpu(force = None):
    if force is not None:return force
    s = os.popen("nvidia-smi --query-gpu=memory.free --format=csv")
    a = []
    ss = s.read().replace('MiB', '').replace('memory.free', '').split('\n')
    s.close()
    for i in range(1, len(ss) - 1):
        a.append(int(ss[i]))
    if len(a) == 0: return torch.device('cpu')
    best = int(np.argmax(a))
    print('the best GPU is ',best,' with free memories of ',ss[best + 1])
    return torch.device('cuda:'+str(best))

def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))