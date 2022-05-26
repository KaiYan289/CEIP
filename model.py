import torch.nn as nn
from layers import *
import os
import numpy as np
import wandb
class FlowModel(nn.Module): 
    def __init__(self, action_size, obs_size, task_num, typ, env, seed=21921974):
        super().__init__()
        assert env in ["fetchreach", "kitchen", "office"], "unknown architecture!"
        #torch.manual_seed(seed) # REMEMBER TO REMOVE THIS WHEN THE CODE IS READY!
        #np.random.seed(seed)
        self.action_size = action_size
        self.task_num = task_num
        self.gauss = torch.distributions.MultivariateNormal(torch.zeros(self.action_size), torch.eye(self.action_size)) # not eye?
        
        if typ == "1layer_debug": # with explicit prior
            self.model = nn.ModuleList([Parallel_Actnorms(action_size, task_num, obs_size=obs_size * 2, env=env)])
        elif typ == "1layer_single": # without explicit prior
            self.model = nn.ModuleList([Parallel_Actnorms(action_size, task_num, obs_size=obs_size, env=env)])
        
    def predict(self, state, task_idx):
        z = self.gauss.sample((state.shape[0], )).to('cuda:0') # shape[0] is batch_size
        for i in range(len(self.model)):
            z = self.model[i].forward(z, task_idx, obs=state)
        return z
        # input state, output action distribution

    def forward_pass(self, z0, state, task_idx):
        z = z0.clone()
        for i in range(len(self.model)):
            z = self.model[i].forward(z, task_idx, obs=state)
        return z

    def inverse_forward(self, x, state, task_idx):
        for i in reversed(range(len(self.model))):
            x = self.model[i].forward(x, task_idx, obs=state, inverse=True)[1]

        
    def get_log_prob(self, z0, state, task_idx):
        tot_logdet = torch.zeros(1, z0.shape[0]).to('cuda:0')
        z = z0.clone() # DON'T FORGET THIS CLONE!!!
        for i in reversed(range(len(self.model))):
            logdet, z = self.model[i].forward(z, task_idx, obs=state, inverse=True)
            tot_logdet += logdet
        
        if task_idx < self.task_num:
            return torch.mean(tot_logdet.squeeze()) - torch.mean(z.square().sum(axis=1) / 2) - z.shape[1] / 2 * math.log(2 * math.pi), \
               (torch.mean(tot_logdet.squeeze()).detach(), - torch.mean(z.square().sum(axis=1) / 2).detach(), -z.shape[1] / 2 * math.log(2 * math.pi))
        else: 
            return torch.mean(tot_logdet.squeeze()) - torch.mean(z.square().sum(axis=1) / 2) - z.shape[1] / 2 * math.log(2 * math.pi), \
               (torch.mean(tot_logdet.squeeze()).detach(), - torch.mean(z.square().sum(axis=1) / 2).detach(), -z.shape[1] / 2 * math.log(2 * math.pi))
                      
    def save(self, num_epoch, name_prefix, name):
        print("name_prefix:", name_prefix)
        print("name:", name)
        name = name_prefix + name + ".pth"
        if not os.path.exists(name_prefix):
            os.mkdir(name_prefix) 
        torch.save(self.model, name)

    def load(self, name):
        name = name
        assert os.path.exists(name), "The file does not exist!"
        self.model = torch.load(name)

