                             # these are ordered dicts where the key : value
                            # is env_name : env_constructor
import numpy as np
import gym
import random
import roboverse
from stable_baselines3 import SAC, PPO
import stable_baselines3
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines import SAC
from model import FlowModel
from stable_baselines3.common.callbacks import CheckpointCallback
import imageio
import os, time
import torch
from stable_baselines3.common.logger import configure
from typing import Callable
device = torch.device("cuda:0")
mode = "train"
# changed from local
import argparse


def get_args():
    global project_name
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=1000009)
    parser.add_argument("--modelseed", help="modelseed", type=int, default=1000009)
    parser.add_argument("--method", help="method", type=str, default="ours_decoupled") # naked, 1layer_mixture
    parser.add_argument("--name", help="name", type=str, default="fetch_hidden")
    parser.add_argument("--TL", help="task list", type=int, default=0)
    parser.add_argument("--n_epoch", help="n_epoch", type=int, default=60)
    parser.add_argument("--n_steps", help="n_steps", type=int, default=4096)
    # 0 = ['microwave', 'kettle', 'top burner', 'light switch']
    # 1 = ['microwave', 'kettle', 'hinge', 'slide']
    args = parser.parse_args()
    print("args:", args)
    return args

def forward_pass(model, z0, state, fix_model=-1):
        task_idx = 99 if fix_model == -1 else fix_model
        z = z0.clone()
        #print("z0:", z0)
        for i in range(len(model)):
            z = model[i].forward(z, task_idx, obs=state)
        #print("z:", z)
        return z

def reverse_pass(model, z0, state, fix_model=-1):
        task_idx = 99 if fix_model == -1 else fix_model
        z = z0.clone()
        tot_logdet = torch.zeros(1, z.shape[0]).to(device)
        print("z0:", z0)
        for i in reversed(range(len(model))):
            logdet, z = model[i].forward(z, task_idx, obs=state, inverse=True)
            tot_logdet += logdet
        print("z:", z)
        return z

reward_counts = []

class Wrapper_Env_Hard(gym.Env):
    def __init__(self, inner_env, args):
        self.inner_env = inner_env
        self.args_type = args.method
        self.inner_env._max_episode_step = 350
        self.IMG = []
        if self.args_type.find("naked") == -1:
            self.action_space = gym.spaces.Box(low=-3 * np.ones(8), high=3 * np.ones(8))#  # gym.spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]))
        else:
            self.action_space = inner_env.action_space
        print(inner_env.observation_space, inner_env.action_space)
        #exit(0)
        self.observation_space = gym.spaces.Box(low=-np.inf * np.ones(97), high=np.inf * np.ones(97)) # 
        # will this np.inf->np.ones make a difference?
        if mode != "train":
            self.f = open("debug_"+str(args.method)+".txt","w")
        print("method:", args.method)
        
        if self.args_type == "ours_decoupled":
            if args.TL == 0: self.pretrained_model = torch.load("demo/ours/office/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_office_24_TS-sub_transfer.pth").to(device).double()   # ours-noTS
            elif args.TL == 1: self.pretrained_model = torch.load("demo/ours/office/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_office_25_TS-sub_transfer.pth").to(device).double() # ours-withTS
            elif args.TL == 2: self.pretrained_model = torch.load("demo/PARROT/office/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_office_1PARROT_decoupled_noEX_TS-sub_train.pth").to(device).double() # PARROT-TS
            elif args.TL == 3: self.pretrained_model = torch.load("demo/PARROT/office/models/1layer_single_seed"+str(args.modelseed)+"_decoupled_phase_withBN_office_1PARROT_decoupled_noEX_TA_train.pth").to(device).double() # PARROT-TA
            #self.pretrained_model = torch.load("models/pretrain_1layer_debug_fetchhard_hidden_seed"+str(args.modelseed)+"_decoupled_phase_transfer.pth").to(device).double()
            self.pretrained_model.eval()
            """
            ####################
            traj = torch.load("../data/TS.pt")
            if len(traj) > 1:
                traj[0]["observations"] = np.concatenate([traj[i]["observations"] for i in range(len(traj))], axis=0)
                traj[0]["actions"] = np.concatenate([traj[i]["actions"] for i in range(len(traj))], axis=0)
    
            idx = torch.randperm(traj[0]["observations"].shape[0])
            p = len(idx)
            print(len(traj), traj[0]["observations"].shape[0], traj[0]["actions"].shape[0])
            feature = torch.cat([torch.from_numpy(traj[0]["observations"][idx[i]]).unsqueeze(0) for i in range(p)], dim=0).to(device).double()
            label = torch.cat([torch.from_numpy(traj[0]["actions"][idx[i]]).unsqueeze(0) for i in range(p)], dim=0).to(device).double()

            ans = reverse_pass(self.pretrained_model, label, feature)
            gauss_numerator = - torch.mean(ans.square().sum(axis=1) / 2).detach()
            for i in range(ans.shape[0]):print(ans[i])
            print("abs>4:", (ans.abs() > 3).sum(),"absmean:", ans.abs().mean(),"absstd:", ans.abs().std(),"shape:", ans.shape,"ansmax:", ans.max(),"ansmin:", ans.min(), "GN:", gauss_numerator)
            exit(0)
            ####################
            """
            #for param in self.pretrained_model.parameters():
            #    param.requires_grad = False
        elif self.args_type != "naked":
            raise NotImplementedError("Error!")
        print(self.observation_space)
        self.step_count = 0
        
    def reset(self):
        self.step_count = 0
        steps = self.inner_env.reset()
        steps = steps["state"] 
        self.last_obs = steps
        #print("obs:",self.last_obs.shape)
        # exit(0)
        return steps # return the beginning point of the last step
        
    def render(self):
        obs = np.transpose(self.inner_env.render_obs(), (1, 2, 0))
        print(obs.shape)
        return obs
    
    def step(self, action):
        if mode != "train":
            for i in range(action.shape[0]):self.f.write(str(action[i])+" ")
            self.IMG.append(self.render())
            # action =  np.random.normal(size=9)#np.random.random(size=9) * 2 - 1 # # #np.zeros(9)
        # print(self.last_obs)

        t0 = time.time()

        action_new = torch.from_numpy(action).view(1, -1).to(device).clone()
        action_new = forward_pass(self.pretrained_model, action_new, torch.from_numpy(self.last_obs).view(1, -1).to(device)).cpu().detach().numpy().reshape(-1)

        # print("step!", self.step_count)
        t1 = time.time()
        # self.inner_env.render()
        if mode != "train":
            self.f.write(str(action_new[0])+" "+str(action_new[1])+" "+str(action_new[2])+" "+str(action_new[3])+"\n")
            self.f.flush()
        steps = list(self.inner_env.step(action_new))
        steps[0] = steps[0]["state"]
        t2 = time.time()
        # print("done:", steps[2])
        #print("agent action time:", t1 - t0, "env step time:", t2 - t1)
        # print("steps:", steps)
        self.last_obs = steps[0]
        # steps[-1] = {} # drop redundant messages
        self.step_count += 1
        steps[2] = steps[2] or self.step_count >= 350
        # print(self.step_count)
        return steps
        
if __name__ == "__main__":
    args = get_args()
    
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed) # when using multiple GPUs 
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed) 
    random.seed(args.seed) 
    torch.backends.cudnn.deterministic = True 
    # torch.use_deterministic_algorithms(True) use with caution; this line of code changes many behavior of program. 
    torch.backends.cudnn.benchmark = False # CUDNN will try different methods and use an optimal one if this is set to true. This could be harmful if your input size / architecture is changing.
    
    
    simple_name = "office"
    
    env0 = gym.make('Widow250OfficeFixed-v0')#gym.make("FetchReach-v1")
    env0.seed(args.seed)
    env = Wrapper_Env_Hard(env0, args)
    # pretraining_prior = True
    # if pretraining_prior:
     # env = BasicEnv(env, args.seed, args.method, simple_name)  # pretraining prior on
    # env = SubprocVecEnv([env for _ in range(4)]) 
    name = str(args.n_steps)+"stp_"+str(args.n_epoch)+"nepoch_"+args.method+"_modelseed"+str(args.modelseed)+"_seed"+str(args.seed)+"_"+str(args.TL)+"_PPO"

    # os.system("cp RL_PPO_formal.py code_backup/RL_PPO_formal_"+str(time.strftime("%a-%b-%d-%H:%M:%S-%Y", time.localtime()))+".py") 
    
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path="RL_models", name_prefix=name)
    
    #model = PPO("MlpPolicy", env, verbose=2, batch_size=256)
    
     
    #model.set_logger(configure("./log/"+name, ["stdout", "csv", "tensorboard"]))
    # policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Tanh, net_arch=[64, 64]) if args.wdk == 0 else dict(activation_fn=torch.nn.modules.activation.Tanh, net_arch=[64, 64], optimizer_kwargs=dict(weight_decay=args.wdk))
    model = PPO("MlpPolicy", env, verbose=2, tensorboard_log="./log/"+args.method+"_"+str(args.seed)+"_"+str(args.modelseed), n_epochs=args.n_epoch, seed=args.seed, n_steps=args.n_steps) 
    model.set_logger(configure("./log/"+name, ["stdout", "csv", "tensorboard"]))
    print("learn!")
    # 
    if mode == "train": 
        env.reset()
        """
        for i in range(3):
            action, _states = model.predict(env.last_obs)
            print(action)
        exit(0)
        """
        model.learn(total_timesteps=2000000, log_interval=1,  callback=checkpoint_callback) 
        model.save("RL_model_alones/"+name)
    else: 
        #lst = [param.clone() for param in env.pretrained_model.parameters()]
        print(model.__dict__)
        #exit(0) 
        model.learn(total_timesteps=700, log_interval=1)
        imageio.mimsave('eval_'+name+'.mp4', env.IMG, fps=25)
    """
    obs, tot_traj = env.reset(), 0 
    avg_reward = 0 
    lst = []
    while tot_traj < 200: # generate 200 trajectories
        lst.append({})
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        avg_reward += reward
        if done:
          print(tot_traj)
          obs = env.reset()
          tot_traj += 1
    
    print("average reward:", avg_reward / 200)
    f=open(args.method+"_metaworld_"+str(args.seed)+".txt","a")
    f.write(str(avg_reward / 200)+"\n")
    f.close()
    
    python3 RL_fetch.py --seed 100007 --method "naked" --name "fetch"
    python3 RL_fetch.py --seed 100008 --method "naked" --name "fetch"
    python3 RL_fetch.py --seed 100009 --method "naked" --name "fetch"
    
    
    """
    
