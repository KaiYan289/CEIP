import numpy as np
import gym
from stable_baselines3 import SAC
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines import SAC
from model import FlowModel
import os, time
import torch
from stable_baselines3.common.logger import configure

device = torch.device("cuda:0")
mode = "train"

import argparse

def get_args():
    global project_name
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=1000009)
    parser.add_argument("--modelseed", help="modelseed", type=int, default=1000009)
    parser.add_argument("--method", help="method", type=str) 
    parser.add_argument("--name", help="name", type=str, default="fetch_hidden")
    parser.add_argument("--trainsize", help="trainsize", type=int)
    parser.add_argument("--transfersize", help="transfersize", type=int)
    parser.add_argument("--direction", help="direction", type=float)
    args = parser.parse_args()
    print("args:", args)
    return args

def forward_pass(model, z0, state, fix_model=-1):
        task_idx = 8 if fix_model == -1 else fix_model
        z = z0.clone()
        for i in range(len(model)):
            z = model[i].forward(z, task_idx, obs=state)
        return z

class Wrapper_Env_Hard(gym.Env):
    def __init__(self, inner_env, args):
        self.inner_env = inner_env
        self.args_type = args.method
        if self.args_type.find("naked") == -1:
            self.action_space = gym.spaces.Box(low=-3 * np.ones(4), high=3 * np.ones(4))#  # gym.spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]))
        else:
            self.action_space = inner_env.action_space
            
        self.observation_space =  gym.spaces.Box(low=-np.inf * np.ones(10), high=np.inf * np.ones(10)) # inner_env.observation_space
        if mode != "train":
            self.f = open("debug_"+str(args.direction)+"_"+str(args.method)+".txt","w")
        print("method:", args.method)
        
        if self.args_type == "ours": # avoiding strange chars (e.g. \r)
            self.pretrained_model = torch.load("demo/ours/fetchreach/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_fetchreach_8_fetchhard_hidden_"+str(args.direction)+"_transfer.pth").to(device).double()
        elif self.args_type == "ours_related":
            self.pretrained_model = torch.load("demo/ours/fetchreach/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_fetchreach_2_fetchhard_hidden_"+str(args.direction)+"_transfer.pth").to(device).double()
        elif self.args_type == "ours_fourwayrelated":
            self.pretrained_model = torch.load("demo/ours/fetchreach/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_fetchreach_4_fetchhard_hidden_"+str(args.direction)+"_transfer.pth").to(device).double()
        elif self.args_type == "ours_withTS":
            self.pretrained_model = torch.load("demo/ours/fetchreach/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_fetchreach_9_fetchhard_hidden_"+str(args.direction)+"_transfer.pth").to(device).double()
        ########################
        elif self.args_type == "alone_all":
            self.pretrained_model = torch.load("demo/PARROT/fetchreach/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_fetchreach_1BCFLOW_all_fetchhard_hidden_"+str(args.direction)+"_train.pth").to(device).double()
        elif self.args_type == "alone_allwithoutDdemo":
            self.pretrained_model = torch.load("demo/PARROT/fetchreach/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_fetchreach_1BCFLOW_allwithoutDdemo_fetchhard_hidden_"+str(args.direction)+"_train.pth").to(device).double()
        elif self.args_type == "alone_Ddemoonly":
            self.pretrained_model = torch.load("demo/PARROT/fetchreach/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_fetchreach_1BCFLOW_Ddemoonly_fetchhard_hidden_"+str(args.direction)+"_train.pth").to(device).double() 
        elif self.args_type == "alone_relatedwithDdemo":
            self.pretrained_model = torch.load("demo/PARROT/fetchreach/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_fetchreach_1BCFLOW_relatedwithDdemo_fetchhard_hidden_"+str(args.direction)+"_train.pth").to(device).double()
        elif self.args_type == "alone_relatedwithoutDdemo":
            self.pretrained_model = torch.load("demo/PARROT/fetchreach/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_fetchreach_1BCFLOW_relatedwithoutDdemo_fetchhard_hidden_"+str(args.direction)+"_train.pth"").to(device).double()
        elif self.args_type == "alone_fourwayrelatedwithDdemo":
            self.pretrained_model = torch.load("demo/PARROT/fetchreach/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_fetchreach_1BCFLOW_fourwayrelatedwithDdemo_fetchhard_hidden_"+str(args.direction)+"_train.pth").to(device).double()
        print(self.pretrained_model)
        print(self.observation_space)
        self.step_count = 0
        # print(self.pretrained_model)
    def reset(self):
        self.inner_env.reset() # hack: gym does not allow step() before reset()
        self.step_count = 0
        steps = self.inner_env.reset_with_index(args.direction)
        # print("steps:", steps)
        stp = np.random.randint(5, 20)
        env0._max_episode_steps = stp + 40
        random_action = (np.random.random(4) * 2 - 1)
        for i in range(stp):
            # random_action = (np.random.random(4) * 2 - 1)
            steps = self.inner_env.step(random_action)
        steps = steps[0]["observation"]# np.concatenate([steps["observation"], steps["desired_goal"]], axis=-1)
        self.last_obs = steps
        return steps # return the beginning point of the last step
        
    def step(self, action):
        # print(self.last_obs[:3])
        if mode != "train":
            for i in range(self.last_obs.shape[0]):self.f.write(str(self.last_obs[i])+" ")
            self.f.write(str(action[0])+" "+str(action[1])+" "+str(action[2])+" ")
        if self.args_type in ["ours"]:
            action_new = torch.from_numpy(action).view(1, -1).to(device).clone()
            action_new = forward_pass(self.pretrained_model, action_new, torch.from_numpy(self.last_obs).view(1, -1).to(device)).cpu().detach().numpy().reshape(-1)
        elif self.args_type in ["ours_related"]:
            action_new = torch.from_numpy(action).view(1, -1).to(device).clone()
            action_new = forward_pass(self.pretrained_model, action_new, torch.from_numpy(self.last_obs).view(1, -1).to(device), fix_model=2).cpu().detach().numpy().reshape(-1)
        elif self.args_type in ["ours_fourwayrelated"]:
            action_new = torch.from_numpy(action).view(1, -1).to(device).clone()
            action_new = forward_pass(self.pretrained_model, action_new, torch.from_numpy(self.last_obs).view(1, -1).to(device), fix_model=4).cpu().detach().numpy().reshape(-1)
        elif self.args_type in ["ours_withTS"]:
            action_new = torch.from_numpy(action).view(1, -1).to(device).clone()
            action_new = forward_pass(self.pretrained_model, action_new, torch.from_numpy(self.last_obs).view(1, -1).to(device), fix_model=9).cpu().detach().numpy().reshape(-1)
        elif self.args_type.find("alone") != -1: # PARROT
            action_new = torch.from_numpy(action).view(1, -1).to(device).clone()
            action_new = forward_pass(self.pretrained_model, action_new, torch.from_numpy(self.last_obs).view(1, -1).to(device), fix_model=0).cpu().detach().numpy().reshape(-1)
        else:
            action_new = action
        if mode != "train":
            self.f.write(str(action_new[0])+" "+str(action_new[1])+" "+str(action_new[2])+" "+str(action_new[3])+"\n")
            self.f.flush()
        steps = list(self.inner_env.step(action_new))
        steps[0] = steps[0]["observation"] # np.concatenate([steps[0]["observation"], steps[0]["desired_goal"]], axis=-1)
        self.last_obs = steps[0]
        # print(steps)
        self.step_count += 1
        return steps
         
if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    simple_name = "fetch"
    
    env0 = gym.make("FetchReach-v1")

    env = Wrapper_Env_Hard(env0, args)
    
    # pretraining_prior = True
    # if pretraining_prior:
     # env = BasicEnv(env, args.seed, args.method, simple_name)  # pretraining prior on
    # env = SubprocVecEnv([env for _ in range(4)]) 
    
    model = SAC("MlpPolicy", env, verbose=2, batch_size=256, learning_starts=1000)
    model.set_logger(configure("./log/reproduce"+args.method+"_"+str(args.direction)+"_modelseed"+str(args.modelseed)+"_seed"+str(args.seed)+"_phase_transfer_"+mode, ["stdout", "csv", "tensorboard"]))
    # model = SAC(MlpPolicy, env, verbose=2, batch_size=1000, tensorboard_log="./log/"+args.method+"_"+str(args.seed))
    print("learn!")
    if mode == "train": 
        model.learn(total_timesteps=30000, log_interval=1) 
        model.save("RL_model_alones/"+args.method+"_"+str(args.direction)+"_modelseed"+str(args.modelseed)+"_seed"+str(args.seed)+"_phase_transfer_"+mode)
    else: 
        model.learn(total_timesteps=1000, log_interval=1)

    
