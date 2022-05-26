import wandb
import sys 
sys.path.append('../../..')
from trainer_ours_forall import Flowtrainer
from hyperparams.kitchen_FIST import *
import os
import time
from datetime import date

def setenv(env_name, statedim, actiondim, tasknum):
    hps_env_setter("env_name", env_name)
    hps_env_setter("state", statedim)
    hps_env_setter("action", actiondim)
    hps_env_setter("task_num", tasknum)

if __name__ == "__main__":
    args = get_args()
    # For readers: substitute the following [two lines to your own.
    real_names = [args.alphabet+"_sub"]
    real_names_abbr = ["formal"]
    wandb.login(key=XXXXXXX)
    wandb.init(entity=XXXXXXX, project="integration_2", name=args.alphabet+"_FIST_ours_decoupled_"+str(args.seed)) 
    # wandb.define_metric("test*", step_metric="valid_step")
    # code backup
    now = time.strftime("%a-%b-%d-%H:%M:%S-%Y", time.localtime())
    save_path = "code_backup/"+str(now)+project_name
    os.mkdir(save_path)
    os.system("cp *.py "+save_path)
    
    # torch.autograd.set_detect_anomaly(True)
    # ep = {"fetchhard_hidden_4.5": 20, "fetchhard_hidden_5.5": 20, "fetchhard_hidden_6.5": 50, "fetchhard_hidden_7.5": 50}
    
    args.train_size = 0
    args.transfer_size = 0
    hps_train_setter("early_stop_num_pretrain", 4000)
    hps_train_setter("early_stop_num_transfer", 4000) 
    hps_train_setter("current_method", "ours_decoupled")
    hps_train_setter("train_epoch", 1000)
    hps_train_setter("transfer_epoch", 1000)
    hps_train_setter("batch_size_train", 128)
    hps_train_setter("batch_size_transfer", 128)
    hps_train_setter("wdk", 0)
    hps_train_setter("seed", args.seed)
    hps_env_setter("alphabet", args.alphabet)
    hps_model_setter("type", "1layer_single") # 1layer_debug for database / 1layer_single for no database
    
    for name in real_names:
        print(name)
        args.env_name = name
        if name != real_names[0]:
            args.train = 0
        else: args.train = 1
        print("args.env_name:", args.env_name)
        # project_name_setter(args.env_name)
        hps_env_setter("env_name", args.env_name)
        # hps_train_setter("transfer_epoch", ep[name])
        print("project_name:", hps_env["env_name"])
        # exit(0)
        setenv(args.env_name, 60, 9, 24)
        Flowtrainer(args, hps_env, hps_train, hps_model)
    """
    hps_model_setter("type", "realnvp")      
    
    for name in real_names:4
        print(name)
        args.env_name = name
        if name != real_names[0]: args.train = 0
        else: args.train = 1
        # args.train = 0
        print("args.env_name:", args.env_name)
        # project_name_setter(args.env_name)
        hps_env_setter("env_name", args.env_name)
        print("project_name:", hps_env["env_name"])
        # exit(0)
        Flowtrainer(args)
    """
