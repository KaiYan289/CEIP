                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor
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
from d4rl.kitchen.kitchen_envs import KitchenBase

device = torch.device("cuda:0")
mode = "train"
# changed from local
import argparse

def get_args():
    global project_name
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=19)
    parser.add_argument("--modelseed", help="modelseed", type=int, default=1000009)
    parser.add_argument("--TL", help="task list", type=int)
    parser.add_argument("--arch", help="arch", type=str, default="3256")
    parser.add_argument("--batchsize", help="batch size", type=int, default=64)
    parser.add_argument("--RL_arch", help="RL arch", type=str, default="64_64")
    parser.add_argument("--target_update", help="target_update", type=float, default=None)
    parser.add_argument("--max_grad_norm", help="max grad norm", type=float, default=0.5)
    parser.add_argument("--gae_lambda", help="gae_lambda", type=float, default=0.95)
    parser.add_argument("--n_steps", help="n_steps", type=int, default=4096)
    parser.add_argument("--n_epochs", help="n_epochs", type=int, default=60)
    args = parser.parse_args()
    print("args:", args)
    return args

def forward_pass(model, z0, state, fix_model=-1):
        task_idx = 99 if fix_model == -1 else fix_model
        z = z0.clone()
        # print("z0:", z0)
        # print(state.shape)
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
            
        if args.TL == 4: self.pretrained_model = torch.load("demo/ours/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_24_A_sub_transfer.pth").to(device).double() 
        elif args.TL == 5: self.pretrained_model = torch.load("demo/ours/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_24_B_sub_transfer.pth").to(device).double()
        elif args.TL == 6: self.pretrained_model = torch.load("demo/ours/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_24_C_sub_transfer.pth").to(device).double()
        elif args.TL == 7: self.pretrained_model = torch.load("demo/ours/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_24_D_sub_transfer.pth").to(device).double()
        elif args.TL == 14: self.pretrained_model = torch.load("demo/ours/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_25_A_sub_transfer.pth").to(device).double() 
        elif args.TL == 15: self.pretrained_model = torch.load("demo/ours/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_25_B_sub_transfer.pth").to(device).double()
        elif args.TL == 16: self.pretrained_model = torch.load("demo/ours/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_25_C_sub_transfer.pth").to(device).double()
        elif args.TL == 17: self.pretrained_model = torch.load("demo/ours/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_25_D_sub_transfer.pth").to(device).double()
        elif args.TL == 24: self.pretrained_model = torch.load("demo/PARROT/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_1PARROT_decoupled_noEX_A_sub_train.pth").to(device).double()
        elif args.TL == 25: self.pretrained_model = torch.load("demo/PARROT/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_1PARROT_decoupled_noEX_B_sub_train.pth").to(device).double()
        elif args.TL == 26: self.pretrained_model = torch.load("demo/PARROT/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_1PARROT_decoupled_noEX_C_sub_train.pth").to(device).double()
        elif args.TL == 27: self.pretrained_model = torch.load("demo/PARROT/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_1PARROT_decoupled_noEX_D_sub_train.pth").to(device).double()
        elif args.TL == 34: self.pretrained_model = torch.load("demo/PARROT/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_1PARROT_decoupled_noEX_kitchen-mixed-no-topknob_train.pth").to(device).double()
        elif args.TL == 35: self.pretrained_model = torch.load("demo/PARROT/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_1PARROT_decoupled_noEX_kitchen-mixed-no-microwave_train.pth").to(device).double()
        elif args.TL == 36: self.pretrained_model = torch.load("demo/PARROT/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_1PARROT_decoupled_noEX_kitchen-mixed-no-kettle_train.pth").to(device).double()
        elif args.TL == 37: self.pretrained_model = torch.load("demo/PARROT/kitchen_FIST/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_kitchen-FIST_1PARROT_decoupled_noEX_kitchen-mixed-no-slide_train.pth").to(device).double()
        
        #self.pretrained_model = torch.load("models/pretrain_1layer_debug_fetchhard_hidden_seed"+str(args.modelseed)+"_decoupled_phase_transfer.pth").to(device).double()
        print(self.pretrained_model)
        self.pretrained_model.eval()
        """
        ####################
        traj = torch.load("data/kitchen-v2/"+MODEL[args.TL]+".pt")
        if len(traj) > 1:
            traj[0]["observations"] = np.concatenate([traj[i]["observations"] for i in range(len(traj))], axis=0)
            traj[0]["actions"] = np.concatenate([traj[i]["actions"] for i in range(len(traj))], axis=0)

        idx = torch.randperm(traj[0]["observations"].shape[0])
        p = len(idx)
        print(len(traj), traj[0]["observations"].shape[0], traj[0]["actions"].shape[0])
        feature = torch.cat([torch.from_numpy(traj[0]["observations"][idx[i]]).unsqueeze(0) for i in range(p)], dim=0).to(device).float()
        label = torch.cat([torch.from_numpy(traj[0]["actions"][idx[i]]).unsqueeze(0) for i in range(p)], dim=0).to(device).float()

        ans = reverse_pass(self.pretrained_model, label, feature)
        gauss_numerator = - torch.mean(ans.square().sum(axis=1) / 2).detach()
        print((ans.abs() > 4).sum(), ans.abs().mean(), ans.abs().std(), ans.shape, ans.max(), ans.min(), "GN:", gauss_numerator)
        exit(0)
        ####################
        """
        self.TL = args.TL
        if args.TL in [4, 14, 24, 34]: data = torch.load("data/kitchen_FIST/24task-A/A_sub.pt")
        elif args.TL in [5, 15, 25, 35]:data = torch.load("data/kitchen_FIST/24task-B/B_sub.pt")
        elif args.TL in [6, 16, 26, 36]: data = torch.load("data/kitchen_FIST/24task-C/C_sub.pt")
        elif args.TL in [7, 17, 27, 37]: data = torch.load("data/kitchen_FIST/24task-D/D_sub.pt")
        """
        for traj in data:
            for i in range(traj["observations"].shape[0] - 1):
                self.key.append(traj["observations"][i].reshape(1, -1))
                self.value.append(traj["observations"][i + 1].reshape(1, -1)) 
        self.key, self.value = np.concatenate(self.key, axis=0), np.concatenate(self.value, axis=0)
        """
        print(self.observation_space)
        self.step_count = 0
        
    def reset(self):
        self.step_count = 0
        steps = self.inner_env.reset()
        self.last_obs = steps
        # print(self.last_obs)
        return steps # return the beginning point of the last step
        
    def render(self):
        obs = self.inner_env.render(mode="rgb_array")
        # print(obs.shape)
        return obs
    def step(self, action):
        if mode != "train":
            self.IMG.append(self.render())
            # action =  np.random.normal(size=9)#np.random.random(size=9) * 2 - 1 # # #np.zeros(9)
        # print(self.last_obs)
        t0 = time.time()
         
        action_new = torch.from_numpy(action).view(1, -1).to(device).clone()
        """
        idx = ((self.key - self.last_obs) ** 2).sum(axis=1).argmin()
        obs = np.concatenate([self.last_obs, self.value[idx]], axis=0)
        # print(((self.key - self.last_obs) ** 2).sum(axis=1).min())
        action_new = forward_pass(self.pretrained_model, action_new, torch.from_numpy(obs).view(1, -1).to(device)).cpu().detach().numpy().reshape(-1)
        """     
        # print(((self.key - self.last_obs) ** 2).sum(axis=1).min())
        if self.TL in [4, 5, 6, 7, 14, 15, 16, 17]: action_new = forward_pass(self.pretrained_model, action_new, torch.from_numpy(self.last_obs).view(1, -1).to(device)).cpu().detach().numpy().reshape(-1)
        else: action_new = forward_pass(self.pretrained_model, action_new, torch.from_numpy(self.last_obs).view(1, -1).to(device), fix_model=0).cpu().detach().numpy().reshape(-1)
        #print("action:", action, "action_new:", action_new)
        t1 = time.time()
        # self.inner_env.render()
        steps = list(self.inner_env.step(action_new))
        t2 = time.time()
        # print("done:", steps[2])
        #print("agent action time:", t1 - t0, "env step time:", t2 - t1)
        # print("steps:", steps)
        self.last_obs = steps[0]
        
        # steps[-1] = {} # drop redundant messages
        self.step_count += 1
        steps[2] = steps[2] or self.step_count >= 280
        # print(self.step_count)
        return steps
        
class MKTS(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'top burner', 'light switch']
class MBSS(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'bottom burner', 'light switch', 'slide cabinet']
class MKSH(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet']
class MKHS(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'hinge cabinet', 'slide cabinet'] 



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
    
    if args.TL in [4, 14, 24, 34]: env0 = MKTS()
    elif args.TL in [5, 15, 25, 35]: env0 = MBSS()
    elif args.TL in [6, 16, 26, 36, 7, 17, 27, 37]: env0 = MKSH()
    env0._max_episode_steps = 280
    env0.seed(args.seed)
    env = Wrapper_Env_Hard(env0, args)
    
    t0 = time.time()
    name = "e60_reproduce_tuning_"+str(t0)+"_"+str(args.TL)
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
