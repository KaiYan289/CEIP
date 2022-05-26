import wandb
import sys 
sys.path.append('../../..')
from trainer_ours_forall import Flowtrainer
from hyperparams.fetchreach import *
import os
import time
from datetime import date
#from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            # ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor
#names = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())
# real_names = [name.replace('-goal-observable', '') for name in names]
# real_names = ["plate-slide-v2", "plate-slide-side-v2", "plate-slide-back-v2", "plate-slide-back-side-v2", "push-back-v2", "push-v2", "door-close-v2", "door-lock-v2", "door-open-v2", "door-unlock-v2"] # ["fetch_4.5"]#["fetch_hidden_0", "fetch_hidden_1", "fetch_hidden_2", "fetch_hidden_3", "fetch_hidden_4", "fetch_hidden_5", "fetch_hidden_6", "fetch_hidden_7"]# ["fetch_0", "fetch_1", "fetch_2", "fetch_3", "fetch_4", "fetch_5", "fetch_6", "fetch_7"] # #["fetch_hidden_"]# ["fetch_"]
real_names = ["fetchhard_hidden_4.5", "fetchhard_hidden_5.5", "fetchhard_hidden_6.5", "fetchhard_hidden_7.5"] # ["fetchhard_hidden_4.5", "fetchhard_hidden_5.5", "fetchhard_hidden_6.5", "fetchhard_hidden_7.5"] # ["fetch_hidden_"]#  # ["fetch_0", "fetch_2", "fetch_3", "fetch_4", "fetch_5", "fetch_6", "fetch_7"] #  
# real_names = ["plate-slide-v2", "plate-slide-side-v2", "plate-slide-back-v2", "plate-slide-back-side-v2", "push-back-v2", "push-v2"]# 
# real_names = ["soccer-v2", "door-close-v2", "door-lock-v2", "door-open-v2", "door-unlock-v2"]

def setenv(env_name, statedim, actiondim, tasknum):
    hps_env_setter("env_name", env_name)
    hps_env_setter("state", statedim)
    hps_env_setter("action", actiondim)
    hps_env_setter("task_num", tasknum)

if __name__ == "__main__":
    args = get_args()
    # For readers: substitute the following [two lines to your own.
    wandb.login(key=XXXXXXX)
    wandb.init(entity=XXXXXXX, project="integration", name="BCFLOW_all_"+str(args.seed))
    wandb.define_metric("valid_step")
    # wandb.define_metric("test*", step_metric="valid_step")
    # code backup
    
    args.train_size = 1600
    args.transfer_size = 160
    args.val_interval = 20
    now = time.strftime("%a-%b-%d-%H:%M:%S-%Y", time.localtime())
    save_path = "code_backup/"+str(now)+project_name
    os.mkdir(save_path)
    os.system("cp *.py "+save_path)
    hps_train_setter("train_size", args.train_size)
    hps_train_setter("transfer_size", args.transfer_size)
    hps_train_setter("early_stop_num_pretrain", 100)
    hps_train_setter("train_epoch", 1000)
    hps_train_setter("batch_size_train", 160)
    hps_train_setter("batch_size_transfer", 40)
    hps_train_setter("current_method", "BCFLOW_all")
    hps_train_setter("seed", args.seed)
    
    hps_model_setter("type", "1layer_single")  
    
    
    # ep = {"fetchhard_hidden_4.5": 60, "fetchhard_hidden_5.5": 45, "fetchhard_hidden_6.5": 100, "fetchhard_hidden_7.5": 30}
    
    for name in real_names:
        print(name)
        args.env_name = name
        print("args.env_name:", args.env_name)
        # project_name_setter(args.env_name)
        hps_env_setter("env_name", args.env_name)
        # hps_train_setter("train_epoch", ep[name])
        print("project_name:", hps_env["env_name"])
        # exit(0)
        setenv(args.env_name, 10, 4, 1)
        Flowtrainer(args, hps_env, hps_train, hps_model)
