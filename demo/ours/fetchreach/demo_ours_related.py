import wandb
import sys 
sys.path.append('../../..')
from trainer_ours_forall import Flowtrainer
from hyperparams.fetchreach import *
import os
import time
from datetime import date
real_names = ["fetchhard_hidden_4.5", "fetchhard_hidden_5.5", "fetchhard_hidden_6.5", "fetchhard_hidden_7.5"] 

def setenv(env_name, statedim, actiondim, tasknum):
    hps_env_setter("env_name", env_name)
    hps_env_setter("state", statedim)
    hps_env_setter("action", actiondim)
    hps_env_setter("task_num", tasknum)

if __name__ == "__main__":
    args = get_args()
    # For readers: substitute the following two lines to your own.
    wandb.login(key=XXXXXXX)
    wandb.init(entity=XXXXXXX, project="integration", name="ours_related_"+str(args.seed))
    # wandb.define_metric("test*", step_metric="valid_step")
    # code backup
    now = time.strftime("%a-%b-%d-%H:%M:%S-%Y", time.localtime())
    save_path = "code_backup/"+str(now)+project_name
    os.mkdir(save_path)
    os.system("cp *.py "+save_path)
    
      
    # ep = {"fetchhard_hidden_4.5": 20, "fetchhard_hidden_5.5": 20, "fetchhard_hidden_6.5": 50, "fetchhard_hidden_7.5": 50}
    
    args.train_size = 1600
    args.transfer_size = 160
    hps_train_setter("early_stop_num_pretrain", 1000)
    hps_train_setter("early_stop_num_transfer", 1000) # note: this number should be > 5?
    hps_train_setter("current_method", "ours_related")
    hps_train_setter("train_epoch", 1000)
    hps_train_setter("transfer_epoch", 1000)
    hps_train_setter("batch_size_train", 40)
    hps_train_setter("batch_size_transfer", 40)
    hps_train_setter("seed", args.seed)
    hps_train_setter("train_size", args.train_size)
    hps_train_setter("transfer_size", args.transfer_size)
    hps_model_setter("type", "1layer_single")
    
    for name in real_names:
        print(name)
        args.env_name = name
        args.train = 1 # must be 1; TA is different each time
        print("args.env_name:", args.env_name)
        # project_name_setter(args.env_name)
        setenv(args.env_name, 10, 4, 2)
        Flowtrainer(args, hps_env, hps_train, hps_model)

