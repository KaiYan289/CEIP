# hyperparameters for the training (default values)
import torch.optim
import argparse
import numpy as np
project_name = ""

 
# for our method
"""
hps_train = {
    "train_epoch": 0,
    "transfer_epoch": 1000,
    "optimizer": torch.optim.Adam,
    "lr_pretrain": 0.001,
    "lr_transfer": 0.001, #for constant 0.1
    "batch_size_train": 40,
    "batch_size_transfer": 40,
    "early_stop_num_pretrain": 4000000,
    "early_stop_num_transfer": 4000000,
} 

"""

# for PARROT

hps_train = {
    "current_method": None,
    "train_epoch": None,
    "transfer_epoch": None, # unused
    "optimizer": torch.optim.Adam,
    "lr_pretrain": 0.001,
    "lr_transfer": 0.001, #for constant 0.1
    "n_repeat": 50, # DEPRECATED
    "batch_size_train": None,
    "batch_size_transfer": None,
    "early_stop_num_pretrain": 4000000,
    "early_stop_num_transfer": 128,
    "adapt_batch_size": 128
} 


# hyperparams for the model

hps_model = { # deprecated
    "seed":0,
    "num_flow_layers":3,
    "task_specific_layers":[True, True, True],#[False, False, False, False, False],#
    "flips": [True, True, True],
    "extractor_output_size":256, 
    "flow_middle_size":256,
    "image_input":False,
    "type":None,  
    "rot_pattern": np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8], [4,5,6,7,8,1,2,3,0], [1,2,5,7,4,3,6,8,0], [6, 7, 8, 2, 3, 4, 0, 1, 5]]) # 4-layer
    #
    #np.array([[0, 1, 2, 3], [2, 3, 0, 1], [2, 0, 3, 1], [2, 3, 0, 1]]) # np.array([[2, 1, 0, 3], [3, 0, 2, 1], [1, 0, 3, 2], [1, 2, 0, 3]]) # [None, None, None, None]  # old version: np.array([[0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2], [3, 2, 1, 0]]) # todo: change this dead code.
}
"""

hps_model = {
    "seed":0,
    "num_flow_layers": 1,
    "task_specific_layers":[True],
    "flips":[False],
    "extractor_output_size":256,
    "flow_middle_size":32, 
    "image_input":False,
    "type":None,
    "rot_pattern": None
}
"""
hps_env = {
    # filled by params input.
    "env_name_global": "kitchen-SKiLD"
}

def hps_train_setter(key, val):
    hps_train[key] = val

def hps_model_setter(key, val):
    hps_model[key] = val

def hps_env_setter(key, val):
    hps_env[key] = val

def project_name_setter(val):
    global project_name
    project_name = val

# hyperparams for running configurations
def get_args():
    global project_name
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the config file directory")
    # parser.add_argument("--name", default="skild_ours", help="running name in wandb") 
    parser.add_argument("--seed", default=1000009, type=int, help="random seed") # 19260817
    
    # should be 100009 if not debugging!
    
    parser.add_argument("--prefix", default="models/", help="experiment prefix, if given creates subfolder in experiment directory")
    parser.add_argument('--skip_first_val', default=False, type=int,
                        help='if True, will skip the first validation epoch')
    parser.add_argument('--log_interval', default=1, type=int,
                        help='number of updates per training log')
    parser.add_argument('--val_interval', default=1, type=int,
                        help='number of updates per validation') # evaluate every 40 batches (not epoches!)
    parser.add_argument('--env_name', default="bipedalwalker", help="environment name")
    parser.add_argument('--train', default=1, type=int,help="train or not. 0 no 1 yes.")
    parser.add_argument('--train_size', default=0, type=int,help="train size.") # 1600 for ours; 160 for PARROT
    parser.add_argument('--transfer_size', default=0, type=int,help="transfer size.") # 160 for ours
    parser.add_argument('--first_steps', default=40, type=int, help="first steps.")
    parser.add_argument('--task', default="kitchen-demo-kettle-bottomknob-hinge-slide", type=str, help="demo task")  #
    parser.add_argument("--arch", type=str, help="architecture", default="light")
    parser.add_argument("--alphabet", type=str, help="ABCD")
    args = parser.parse_args()
    project_name_setter(args.env_name)
    print(args)
    return args
