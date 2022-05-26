import numpy as np
import gym
import random
import d4rl.kitchen
from stable_baselines3 import SAC, PPO
import stable_baselines3
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines import SAC

from model import FlowModel
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
import imageio
import os, time
import torch
from stable_baselines3.common.logger import configure

device = torch.device("cuda:0")
mode = "train"
# changed from local
import argparse

def get_args():
    global project_name
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=19)
    parser.add_argument("--modelseed", help="modelseed", type=int, default=1000009)
    parser.add_argument("--TL", help="task list", type=int, default=1)
    parser.add_argument("--arch", help="arch", type=str, default="3256")
    parser.add_argument("--batchsize", help="batch size", type=int, default=64)
    parser.add_argument("--RL_arch", help="RL arch", type=str, default="64_64")
    parser.add_argument("--target_update", help="target_update", type=float, default=None)
    parser.add_argument("--max_grad_norm", help="max grad norm", type=float, default=0.5)
    parser.add_argument("--gae_lambda", help="gae_lambda", type=float, default=0.95)
    parser.add_argument("--n_steps", help="n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", help="n_epochs", type=int, default=60)
    parser.add_argument("--pushforward", help="pushforward", type=str, default="no")
    args = parser.parse_args()
    print("args:", args)
    return args

def forward_pass(model, z0, state, fix_model=-1):
        task_idx = 99 if fix_model == -1 else fix_model
        z = z0.clone()
        # print("z0:", z0)
        for i in range(len(model)):
            z = model[i].forward(z, task_idx, obs=state)
        # print("z:", z)
        return z

def reverse_pass(model, z0, state, fix_model=-1): 
        task_idx = 25 if fix_model == -1 else fix_model
        z = z0.clone()
        tot_logdet = torch.zeros(1, z.shape[0]).to(device)
        print("z0:", z0)
        for i in reversed(range(len(model))):
            logdet, z = model[i].forward(z, task_idx, obs=state, inverse=True)
            tot_logdet += logdet
        print("z:", z)
        return z

class Wrapper_Env_Hard(gym.Env):
    def __init__(self, inner_env, args):
        self.inner_env = inner_env
        self.IMG = []
        
        self.action_space = gym.spaces.Box(low=-3 * np.ones(9), high=3 * np.ones(9))#  # gym.spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]))
        
        print(inner_env.observation_space.low, inner_env.observation_space.high, self.action_space.low, self.action_space.high)
        #exit(0)
        self.observation_space = inner_env.observation_space # gym.spaces.Box(low=-8 * np.ones(60), high=8 * np.ones(60)) # 
        # will this np.inf->np.ones make a difference?
        
        self.push_forward = True if args.pushforward == "yes" else "no"
        self.key, self.value = [], []
        
        if self.push_forward:
            self.num_steps, self.traj_index, self.last_step_on_traj = [], [], []
        
        data = None 
        if args.TL in [100, 200, 230]: data = torch.load("data/kitchen_SKiLD/oneshot-1.pt")
        elif args.TL in [101, 201, 231]: data = torch.load("data/kitchen_SKiLD/oneshot-2.pt")
        
        if self.push_forward:
            self.num_trajs = len(data)
        
        for j, traj in enumerate(data):
            for i in range(traj["observations"].shape[0] - 1):
                self.key.append(traj["observations"][i].reshape(1, -1))
                self.value.append(traj["observations"][i + 1].reshape(1, -1)) 
                if self.push_forward:
                    self.num_steps.append(np.array([i]))
                    self.traj_index.append(j)
            if self.push_forward: self.num_steps = np.concatenate(self.num_steps)
        self.key, self.value = np.concatenate(self.key, axis=0), np.concatenate(self.value, axis=0)
        print(self.key.shape, self.value.shape)
        
        # exit(0)
        if args.TL == 100:
            self.pretrained_model = torch.load("demo/ours/kitchen_SKiLD/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-SKiLD_24_oneshot-1_transfer.pth").to(device).double()  
        elif args.TL == 200:
            self.pretrained_model = torch.load("demo/ours/kitchen_SKiLD/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-SKiLD_25_oneshot-1_transfer.pth").to(device).double()  
        elif args.TL == 101:
            self.pretrained_model = torch.load("demo/ours/kitchen_SKiLD/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-SKiLD_24_oneshot-2_transfer.pth").to(device).double()
        elif args.TL == 201:
            self.pretrained_model = torch.load("demo/ours/kitchen_SKiLD/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-SKiLD_25_oneshot-2_transfer.pth").to(device).double()
        elif args.TL == 230:
            self.pretrained_model = torch.load("demo/PARROT/kitchen_skild/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-SKiLD_1PARROT_ours_decoupled_oneshot-1_train.pth").to(device).double()
        elif args.TL == 231: 
            self.pretrained_model = torch.load("demo/PARROT/kitchen_skild/models/1layer_debug_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-SKiLD_1PARROT_ours_decoupled_oneshot-2_train.pth").to(device).double()
        self.TL = args.TL
        self.pretrained_model.eval()

        print(self.observation_space)
        self.step_count = 0
        
    def reset(self):
        self.step_count = 0
        steps = self.inner_env.reset()
        self.last_obs = steps
        if self.push_forward: self.last_step_on_traj = -np.ones(self.num_trajs)
        # print(self.last_obs)
        return steps # return the beginning point of the last step
        
    def render(self):
        obs = self.inner_env.render(mode="rgb_array")
        # print(obs.shape)
        return obs
    def step(self, action):
        if mode != "train":
            # for i in range(action.shape[0]):self.f.write(str(action[i])+" ")
            self.IMG.append(self.render())
            # action =  np.random.normal(size=9)#np.random.random(size=9) * 2 - 1 # # #np.zeros(9)
        # print(self.last_obs)
        t0 = time.time()
        
        action_new = torch.from_numpy(action).view(1, -1).to(device).clone()
        # print(((self.key - self.last_obs) ** 2).sum(axis=1).min())
        dist = ((self.key - self.last_obs) ** 2).sum(axis=1)
        if self.push_forward:
            for j in range(self.num_trajs):
                dist += (self.traj_index == np.array([j])) * (self.num_steps <= self.last_step_on_traj[j]) # penalty of 1
            idx = dist.argmin()
            self.last_step_on_traj[self.traj_index[idx]] = self.num_steps[idx]
        else: idx = dist.argmin()
        
        obs = np.concatenate([self.last_obs, self.value[idx]], axis=0)
        if self.TL in [100, 101, 200, 201]: action_new = forward_pass(self.pretrained_model, action_new, torch.from_numpy(obs).view(1, -1).to(device)).cpu().detach().numpy().reshape(-1)
        else: action_new = forward_pass(self.pretrained_model, action_new, torch.from_numpy(obs).view(1, -1).to(device), fix_model=0).cpu().detach().numpy().reshape(-1)
        #print("action:", action, "action_new:", action_new)
        t1 = time.time()
        # self.inner_env.render()
        steps = list(self.inner_env.step(action_new))
        t2 = time.time()
        # print("done:", steps[2])
        #print("agent action time:", t1 - t0, "env step time:", t2 - t1)
        # print("steps:", steps)
        self.last_obs = steps[0]
        steps[2] = steps[2] or self.step_count >= 280
        # steps[-1] = {} # drop redundant messages
        self.step_count += 1
        # print(self.step_count)
        return steps
         
if __name__ == "__main__":
    args = get_args()
    
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed) # when using multiple GPUs 
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed) 
    random.seed(args.seed) 
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True 
    # torch.use_deterministic_algorithms(True) use with caution; this line of code changes many behavior of program. 
    torch.backends.cudnn.benchmark = False # CUDNN will try different methods and use an optimal one if this is set to true. This could be harmful if your input size / architecture is changing.
    
    
    simple_name = "kitchen"
    
    if args.TL in [100, 200, 230]: env0 = gym.make('kitchen-kbts-v0')#gym.make("FetchReach-v1")
    else: env0 = gym.make('kitchen-mlsh-v0')
    env0.seed(args.seed)
    env = Wrapper_Env_Hard(env0, args)
    
    t0 = time.time()
    name = "e60_reproduce_tuning_"+str(t0)+"_"+str(args.TL) + ("_vorwarts" if args.pushforward else "")
    f = open("log/conf/"+name+"_conf.txt", "w")
    for arg in vars(args):
        f.write(str(arg)+" "+str(getattr(args, arg))+"\n")
    f.write("filename:" +name)
    f.close()
    checkpoint_callback = CheckpointCallback(save_freq=200000, save_path="RL_models", name_prefix=name) 
    # policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Tanh, net_arch=[64, 64])
    policy_kwargs = {}
    if args.RL_arch == "128_128_64": policy_kwargs={"net_arch": [dict(pi=[128, 128, 64], vf=[128, 128, 64])]}
    elif args.RL_arch == "32_32": policy_kwargs={"net_arch": [dict(pi=[32, 32],vf=[32, 32])]}
    
    model = PPO("MlpPolicy", env, verbose=2, policy_kwargs=policy_kwargs, batch_size=args.batchsize, n_steps=args.n_steps,\
    target_kl=args.target_update, gae_lambda=args.gae_lambda, n_epochs=args.n_epochs, max_grad_norm=args.max_grad_norm, \
    seed=args.seed
    )
    
    torch.set_printoptions(threshold=5000, precision=12)
    """
    for param in model.policy.mlp_extractor.policy_net.parameters():
        print(param)
    exit(0)
    """
    model.set_logger(configure("./log/"+name, ["stdout", "csv", "tensorboard"]))
    print("learn!")
    # 
    if mode == "train": 
        model.learn(total_timesteps=200000, log_interval=1,  callback=checkpoint_callback) 
        model.save("RL_model_alones/"+name)
    else: 
        print(model.__dict__)
        model.learn(total_timesteps=280, log_interval=1)
        imageio.mimsave('eval_'+args.arch+'_'+name+'.mp4', env.IMG, fps=25)
