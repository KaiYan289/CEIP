import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import init_uniform
from functools import reduce
import matplotlib.pyplot as plt
import time

# standard_block
class standard_block(nn.Module):
    def __init__(self, input_size, output_size, use_tanh=False, mode='standard'):
        # normalize is for better initialization.
        super().__init__()
        self.use_tanh = use_tanh
        self.tag = "layers"
        self.output_size = output_size
        if mode == "standard": # single flow of kitchen / office
            middle_size = 256 
            self.net = nn.Sequential(
                nn.Linear(input_size, middle_size),
                nn.BatchNorm1d(middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.BatchNorm1d(middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.BatchNorm1d(middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, output_size)
            )
        elif mode == "simplified": # combination of kitchen / office
            middle_size = 64
            self.net = nn.Sequential(
                nn.Linear(input_size, middle_size),
                nn.BatchNorm1d(middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.BatchNorm1d(middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, output_size)
            )
        else: # fetchreach
            middle_size = 32
            self.net = nn.Sequential(
                   nn.Linear(input_size, middle_size),
                   nn.ReLU(),
                   nn.Linear(middle_size, middle_size),
                   nn.ReLU(),
                   nn.Linear(middle_size, output_size)
            )
    
    def forward(self, x):
        if not self.use_tanh:
            return self.net(x)
        else: return F.tanh(self.net(x))

# parallel_actnorms
class Parallel_Actnorms(nn.Module):
    def __init__(self, size, task_num, obs_size=None, seed=32195821, env="fetchreach"):
        super().__init__()
        assert env in ["fetchreach", "kitchen", "office"], "unknown architecture!"
        
        self.size, self.task_num = size, task_num
        self.env = env
        if env != "fetchreach": 
            self.s, self.b = nn.ModuleList([standard_block(obs_size, size, mode='standard') for i in range(task_num)]), nn.ModuleList([standard_block(obs_size, size, mode='standard') for i in range(task_num)])
            self.importance = standard_block(obs_size, task_num * 2, use_tanh=False, mode="simplified")
            
        else: 
            self.s, self.b = nn.ModuleList([standard_block(obs_size, size, mode='fetchreach') for i in range(task_num)]), nn.ModuleList([standard_block(obs_size, size, mode='fetchreach') for i in range(task_num)])
            self.importance = standard_block(obs_size, task_num * 2, use_tanh=False, mode='fetchreach')
        

    def forward(self, x, task_idx, obs=None, W=None, inverse=False):
        if task_idx >= self.task_num:
            t0 = time.time()

            imp_tmp = self.importance(obs)
            importance = torch.zeros_like(imp_tmp)
            importance[:, self.task_num:] = imp_tmp[:, self.task_num:]
            importance[:, :self.task_num] = F.softplus(imp_tmp[:, :self.task_num]) / self.task_num + (0.0001 if self.env == "fetchreach" else 0.0001) # bounded from below at 0.01 to prevent numerical instability

            t1 = time.time()
          
        if task_idx < self.task_num:
            s, b = torch.exp(self.s[task_idx](obs)), self.b[task_idx](obs)
        else:
            s, b = [], []
            for i in range(self.task_num):
                v1 = torch.exp(self.s[i](obs))
                v2 = self.b[i](obs)
                #print("v1:", v1, "v2:", v2)
                s.append(importance[:, i].unsqueeze(-1) * v1)
                b.append(importance[:, i + self.task_num].unsqueeze(-1) * v2)
            s, b = reduce(lambda z, w: z+w, s), reduce(lambda z, w: z+w, b)

        t2 = time.time()
        if not inverse:
            ans = x * s + b
            t3 = time.time()
            return ans
        else:
            ans = (x - b) / s
            logdet = torch.sum(torch.log(torch.abs(1 / s)), dim=1).view(1, -1)
            return logdet, ans
