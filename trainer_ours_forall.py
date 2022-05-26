import wandb
import torch
# from hyperparams import hps_env, hps_model, hps_train, hps_env_setter, project_name
from utils import AttrDict, get_best_gpu, map_dict
from model import FlowModel

import numpy as np
import torch.nn as nn
import math
from tqdm import tqdm
import copy
import random
from layers import standard_block

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
from functools import reduce
import os
# SMALL_DATA_SIZE, BIG_DATA_SIZE, TEST_DATA_SIZE = 100, 2500, 5000

def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

 
class Flowtrainer:
    def __init__(self, args, hps_env, hps_train, hps_model):
        self.args = args
        self.device = torch.device('cuda:0')# get_best_gpu()
        
        project_name = args.env_name
        
        self.hps_env, self.hps_train, self.hps_model = hps_env, hps_train, hps_model
        
        seed_all(args.seed)
        
        if hps_env["task_num"] != 1: # CEIP 
        
            if hps_env["env_name_global"] == "kitchen-SKiLD":
    
                if hps_env["task_num"] == 24: self.task_list = ["task_noprior_" + str(j) for j in range(hps_env["task_num"])] + [hps_env["env_name"]]
                else: self.task_list = ["task_noprior_" + str(j) for j in range(hps_env["task_num"] - 1)] + [hps_env["env_name"]] + [hps_env["env_name"]]
                self.model = FlowModel(hps_env["action"], hps_env["state"], hps_env["task_num"], hps_model["type"], env="kitchen", seed=args.seed).to(self.device)
                from dataset.kitchen_skild import get_test_dataset_LL
                # from hyperparams.kitchen_SKiLD import hps_env, hps_model, hps_train, hps_env_setter, project_name
                
            elif hps_env["env_name_global"] == "kitchen-FIST":
                if hps_env["task_num"] == 24: self.task_list = ["task_" + str(j) for j in range(hps_env["task_num"])] + [hps_env["env_name"]]
                else: self.task_list = ["task_" + str(j) for j in range(hps_env["task_num"] - 1)] + [hps_env["env_name"]] + [hps_env["env_name"]]
                self.model = FlowModel(hps_env["action"], hps_env["state"], hps_env["task_num"], hps_model["type"], env="kitchen", seed=args.seed).to(self.device)
                from dataset.kitchen_fist import get_test_dataset_LL
                # from hyperparams.kitchen_FIST import hps_env, hps_model, hps_train, hps_env_setter, project_name
            
            elif hps_env["env_name_global"] == "fetchreach":
                if hps_env["task_num"] == 8: # withoutTS 
                    self.task_list = ["fetchhard_hidden_0.0", "fetchhard_hidden_1.0", "fetchhard_hidden_2.0", "fetchhard_hidden_3.0", "fetchhard_hidden_4.0", "fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_7.0", hps_env["env_name"]]
                elif hps_env["task_num"] == 9: # withTS 
                    self.task_list = ["fetchhard_hidden_0.0", "fetchhard_hidden_1.0", "fetchhard_hidden_2.0", "fetchhard_hidden_3.0", "fetchhard_hidden_4.0", "fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_7.0", hps_env["env_name"], hps_env["env_name"]]
                elif hps_env["task_num"] == 2: # related
                    if args.env_name.find("4.5") != -1: self.task_list = ["fetchhard_hidden_4.0", "fetchhard_hidden_5.0", "fetchhard_hidden_4.5"]#["fetchhard_hidden_0.0", "fetchhard_hidden_1.0", "fetchhard_hidden_2.0", "fetchhard_hidden_3.0", "fetchhard_hidden_4.0", "fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_7.0", hps_env["env_name"]] #
                    elif args.env_name.find("5.5") != -1: self.task_list = ["fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_5.5"]#["fetchhard_hidden_0.0", "fetchhard_hidden_1.0", "fetchhard_hidden_2.0", "fetchhard_hidden_3.0", "fetchhard_hidden_4.0", "fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_7.0", hps_env["env_name"]]#
                    elif args.env_name.find("6.5") != -1: self.task_list = ["fetchhard_hidden_6.0", "fetchhard_hidden_7.0", "fetchhard_hidden_6.5"]#["fetchhard_hidden_0.0", "fetchhard_hidden_1.0", "fetchhard_hidden_2.0", "fetchhard_hidden_3.0", "fetchhard_hidden_4.0", "fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_7.0", hps_env["env_name"]]#
                    elif args.env_name.find("7.5") != -1: self.task_list = ["fetchhard_hidden_7.0", "fetchhard_hidden_0.0", "fetchhard_hidden_7.5"]
                elif hps_env["task_num"] == 4: # fourwayrelated
                    if args.env_name.find("4.5") != -1: self.task_list = ["fetchhard_hidden_3.0", "fetchhard_hidden_4.0", "fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_4.5"]
                    elif args.env_name.find("5.5") != -1: self.task_list = ["fetchhard_hidden_4.0", "fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_7.0", "fetchhard_hidden_5.5"]
                    elif args.env_name.find("6.5") != -1: self.task_list = ["fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_7.0", "fetchhard_hidden_0.0", "fetchhard_hidden_6.5"]
                    elif args.env_name.find("7.5") != -1: self.task_list = ["fetchhard_hidden_6.0", "fetchhard_hidden_7.0", "fetchhard_hidden_0.0", "fetchhard_hidden_1.0", "fetchhard_hidden_7.5"]
                    
                self.model = FlowModel(hps_env["action"], hps_env["state"], hps_env["task_num"], hps_model["type"], env="fetchreach", seed=args.seed).to(self.device)
                
                from dataset.fetchreach import get_test_dataset_LL
                # print("!!!!!!!!!!!!!!!!")
                # from hyperparams.fetchreach import hps_env, hps_model, hps_train, hps_env_setter, project_name
            
            elif hps_env["env_name_global"] == "office":
                if hps_env["task_num"] == 24: self.task_list = ["task_" + str(j) for j in range(hps_env["task_num"])] + [hps_env["env_name"]]
                else: self.task_list = ["task_" + str(j) for j in range(hps_env["task_num"] - 1)] + [hps_env["env_name"]] + [hps_env["env_name"]]
                self.model = FlowModel(hps_env["action"], hps_env["state"], hps_env["task_num"], hps_model["type"], env="office", seed=args.seed).to(self.device)
                from dataset.office import get_test_dataset_LL
                # from hyperparams.office import hps_env, hps_model, hps_train, hps_env_setter, project_name
            
            else: raise NotImplementedError("Error!")
            
        else: # PARROT
            self.task_list = [hps_env["env_name"]]
            if hps_env["env_name_global"] == "kitchen-SKiLD":
                self.model = FlowModel(hps_env["action"], hps_env["state"], hps_env["task_num"], hps_model["type"], env="kitchen", seed=args.seed).to(self.device)
                from dataset.kitchen_skild import get_test_dataset_LL
            elif hps_env["env_name_global"] == "kitchen-FIST":
                self.model = FlowModel(hps_env["action"], hps_env["state"], hps_env["task_num"], hps_model["type"], env="kitchen", seed=args.seed).to(self.device)
                from dataset.kitchen_fist import get_test_dataset_LL
            elif hps_env["env_name_global"] == "fetchreach":
                self.model = FlowModel(hps_env["action"], hps_env["state"], hps_env["task_num"], hps_model["type"], env="fetchreach", seed=args.seed).to(self.device)
                from dataset.fetchreach import get_test_dataset_LL
            elif hps_env["env_name_global"] == "office":
                self.model = FlowModel(hps_env["action"], hps_env["state"], hps_env["task_num"], hps_model["type"], env="office", seed=args.seed).to(self.device)
                from dataset.office import get_test_dataset_LL
            else: raise NotImplementedError("Error!")
        
        print("hps_env:", hps_env)
        print("project_name:", project_name)
        print("hps_model:", hps_model)
        print("hps_train:", hps_train)
        
        
        self.model_name = hps_model["type"]+"_seed"+str(self.args.seed)+"_decoupled_phase_withBN_"+str(hps_env["env_name_global"])+"_"+str(hps_env["task_num"]) + ("" if hps_env["task_num"] > 1 else hps_train["current_method"] + "_" + hps_env["env_name"])  # + transfer / train
       
        self.batch_count, self.test_step_count, self.not_improving_count = 0, 0, 0
        self.stop_training = False
        # self.last_vals = []
        
        self.last_val, self.last_rec = 1e10, 0
        self.last_model = None
        
        print("project_name:", project_name)
        if self.args.train == 1:
            self.train_loader, self.val_loader = [], []
            for task_num in range(self.hps_env["task_num"]):
                train_loader, val_loader = get_test_dataset_LL(task_num, self.task_list[task_num], self.args.train_size, self.args.first_steps, self.hps_env, self.hps_train, self.hps_model)
                self.train_loader.append(train_loader)
                self.val_loader.append(val_loader)
        
            self.train(hps_train["train_epoch"])
        else:
            print(self.args.prefix+ self.model_name + "_train.pth")
            self.model.load(self.args.prefix+ self.model_name + "_train.pth")
            # print("models/pretrain_"+str(hps_model["type"])+"_seed"+str(args.seed)+"_decoupled_phase_train_withBN_arch"+self.args.arch+"_v4_"+str(hps_env["task_num"])+".pth")
            # self.model.load("models/FISTized_noprior_"+str(hps_model["type"])+"_seed"+str(args.seed)+"_decoupled_phase_train_withBN_arch"+self.args.arch+"_v4_"+str(hps_env["task_num"])+".pth")
        if hps_env["task_num"] == 1: return
        print("adapt!")
        self.train_query_loader, self.val_query_loader = get_test_dataset_LL(hps_env["task_num"], self.task_list[hps_env["task_num"]], self.args.transfer_size, self.args.first_steps, hps_env, hps_train, hps_model) 

        
        # freeze the parameters
        for name, param in self.model.named_parameters():
            print(name, param.shape)
            if name.find("importance") == -1 and name.find("global_affine") == -1:
                param.requires_grad = False
        
        self.transfer(hps_train["transfer_epoch"]) 
        
    def loss(self, state, label, task_idx):
        if len(task_idx.shape) > 0 and task_idx.shape[0] > 1:
            task_idx = task_idx[0]  # we assert that in a batch all samples are the same.
        # print("state_before:", state[0], "label_before:", label[0])
        res = self.model.get_log_prob(label, state, task_idx)
        # print("state_after:", state[0], "label_after:", label[0])
        # exit(0)
        return (-res[0], res[1])

    def loss_transfer(self, state, label, task_idx):
        if len(task_idx.shape) > 0 and task_idx.shape[0] > 1:
            task_idx = task_idx[0]  # we assert that in a batch all samples are the same.
        res = self.model.get_log_prob(label, state, task_idx)
        return (-res[0], res[1])
        
        
    def transfer(self, epoch): 
        self.model.train()
        self.clear(self.hps_train["lr_transfer"])
        for i in range(epoch):
            begin_time = time.time()
            print("Transfer: Starting epoch", i)
            # self.train_query_loader.dataset.save_datapoint("train_test_for_single")
            # self.val_query_loader.dataset.save_datapoint("test_test_for_single")
            for batch_idx, sample_batched in enumerate(self.train_query_loader):
                
                inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
                # note: this input should have task_number besides spirl's input.
                # "state, action, task_number"
                # input should have "state" and "action" inside.
                t0 = time.time()
                loss, loss_breakdown = self.loss_transfer(inputs["state"], inputs["action"],
                                                 inputs["task_id"])  # should be LSTM with 10 steps?
                
                self.optimizer.zero_grad()
                self.batch_count += 1
                loss.backward()
                # self.model.predict(inputs["state"], hps_env["task_num"]).detach()
                
                nn.utils.clip_grad_norm(self.model.parameters(), 0.0001)
                self.optimizer.step()
                past_time = time.time() - begin_time
                if self.global_batch_idx % self.args.log_interval == 0:
                    # logging outputs...
                    # print("batch_idx:", batch_idx)
                    self.wandb_log("train_transfer", self.batch_count, loss, loss_breakdown)
                if self.global_batch_idx % self.args.val_interval == 0:
                    selected_minn, selected_idx = self.val(self.val_query_loader)
                if self.stop_training: break
                self.global_batch_idx += 1
            if self.stop_training: break
        
        if selected_idx is None:
            selected_idx = self.last_rec #np.array(self.last_vals).argmin()
            selected_minn = self.last_val # np.array(self.last_vals).min()
        wandb.log({"selected_idx": selected_idx, "selected_minn": selected_minn})
        self.model = self.last_model
        
        self.model.save(i, self.args.prefix, self.model_name + "_" +self.hps_env["env_name"]+ "_transfer")

    def clear(self, lr):
        self.last_val, self.last_rec = 1e10, 0
        self.last_model = None
        self.not_improving_count = 0
        self.global_batch_idx = 0
        self.stop_training = False
        self.batch_count, self.test_step_count = 0, 0
        self.optimizer = self.hps_train["optimizer"](self.model.parameters(), lr=lr)

    # specialized for spirl
    def train(self, epoch):
        # if not self.args.skip_first_val:
        #    self.val(self.val_loader)
        self.model.train()
        # wandb.watch(tuple(self.model.model[0].affines.s), log_freq=self.args.log_interval, log="all")
        # T1, T2, T3 = [], [], []
        # self.train_loader, self.val_loader = [], []
        
        print("model:", self.model)
        train_size = len(self.train_loader[0].dataset) # number of samples from each subtask
        for k in range(self.hps_env["task_num"]):
            self.clear(self.hps_train["lr_pretrain"])
            for i in tqdm(range(epoch)):
                for j in range(train_size // self.hps_train["batch_size_train"]):
                    # print(train_size, hps_train["batch_size_train"])
                    sample_batched = next(iter(self.train_loader[k])) # draw a batch from the current task
                    inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
                    loss, loss_breakdown = self.loss(inputs["state"], inputs["action"], inputs["task_id"])
                    # print("\n\nloss:", loss, "loss_breakdown:", loss_breakdown, "\n\n\n\n")
                    self.optimizer.zero_grad()
                    self.batch_count += 1
                    loss.backward()

                    nn.utils.clip_grad_norm(self.model.parameters(), 0.0001)
                    
                    self.optimizer.step()
                    if self.global_batch_idx % self.args.log_interval == 0:
                        # logging outputs...
                        # print("batch_idx:", self.global_batch_idx)
                        self.wandb_log("train", self.batch_count, loss, loss_breakdown)
                    if self.global_batch_idx % self.args.val_interval == 0:
                        selected_minn, selected_idx = self.val(self.val_loader[k])
                        # print("selected_minn:", selected_minn, "selected_idx:", selected_idx)  
                    self.global_batch_idx += 1
                    if self.stop_training: 
                        if selected_idx is None:
                            selected_idx = self.last_rec 
                            selected_minn = self.last_val
                        # wandb.log({"selected_idx": selected_idx, "selected_minn": selected_minn})
                        self.model = self.last_model
                        self.model.save(i, self.args.prefix, "temp"+str(k))
                        break
                if self.stop_training: break
        wandb.log({"selected_idx": selected_idx, "selected_minn": selected_minn})        
        
        # print("finish training")
        
        self.model.save(k, self.args.prefix, self.model_name + "_train")
        
    def val(self, val_loader, test_count=True):
        # run the model on the validation set.
        self.model.eval()
        avg = 0
        tot_loss, tot_breakdown = torch.tensor([0.0]), torch.tensor([0.0, 0.0, 0.0])
        for self.batch_idx, sample_batched in enumerate(val_loader):
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            task_idx = inputs["task_id"][0] # we assert that in a batch all samples are the same.
            loss, loss_breakdown = self.loss(inputs["state"], inputs["action"], task_idx)  # should it be LSTM with 10 steps?
            # TBD: log the loss.
            tot_loss += loss.detach().cpu()
            tot_breakdown[0], tot_breakdown[1], tot_breakdown[2] = tot_breakdown[0] + loss_breakdown[0], tot_breakdown[1] + loss_breakdown[1], tot_breakdown[2] + loss_breakdown[2] 
            avg += loss.detach().cpu()
        if avg / len(val_loader) < self.last_val:
            self.last_rec = self.global_batch_idx
            self.last_model = copy.deepcopy(self.model)
            self.last_val = avg / len(val_loader)
        EARLY_STOP = self.hps_train["early_stop_num_pretrain"] if val_loader.dataset.task_idx < self.hps_env["task_num"] else self.hps_train["early_stop_num_transfer"]
        
        minn, idx = self.last_val, self.last_rec
        
        if self.global_batch_idx > EARLY_STOP and self.stop_training is False:
            if self.last_rec != self.global_batch_idx: # i.e. does not improve
                self.not_improving_count += 1
            else: self.not_improving_count = 0
            if self.not_improving_count >= 0.2 * self.global_batch_idx:
                self.stop_training = True
            
            print("not improving count:", self.not_improving_count)  
            
        if val_loader.dataset.task_idx < self.hps_env["task_num"]: prefix = "test_pretrain"
        else: prefix = "test_transfer_"+self.hps_env["env_name"]
        
        self.wandb_log(prefix+str(val_loader.dataset.task_idx), self.test_step_count, tot_loss / len(val_loader), tot_breakdown / len(val_loader)) # this step will NOT be the training step.
        
        if test_count: self.test_step_count += 1
        self.model.train()

        return minn, idx

    def wandb_log(self, name, idx, loss, loss_breakdown, log_flag=False, other=None):
        # print("wandb log" + name+ "!")
        #norm = 0
        #for param in self.model.parameters():
        #    norm += torch.norm(param)
        if log_flag:
            wandb.log({name + "/batch_idx": self.global_batch_idx,
                   name + "/loss_log": loss.sign() * torch.log10(torch.abs(loss)), name+"_tot_logdet_log": loss_breakdown[0].sign() * torch.log10(torch.abs(loss_breakdown[0])),
                   name + "/gauss_numerator": loss_breakdown[1].sign() * torch.log10(torch.abs(loss_breakdown[1])),
                   name + "/gauss_denominator": loss_breakdown[2] / abs(loss_breakdown[2]) * math.log10(abs(loss_breakdown[2]))})
        else:
            dct = {name + "/batch_idx": self.global_batch_idx,
                   name + "/loss": loss,
                   name + "/tot_logdet": loss_breakdown[0],
                   name + "/gauss_numerator": loss_breakdown[1],
                   name + "/gauss_denominator": loss_breakdown[2],
                   #name + "/debug_norm": norm,
                   }
            if other is not None: dct = dict(dct, **other)
            wandb.log(dct)

