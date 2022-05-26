from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
class Testdataset(Dataset):
    def __init__(self, feature, label, task_idx):
        self.n = feature.shape[0]
        print("n:", self.n)
        self.dim_feature, self.dim_label = feature.shape[1], label.shape[1]
        self.feature = feature
        self.label = label
        self.task_idx = task_idx

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"state": self.feature[idx], "action": self.label[idx], "task_id": self.task_idx}
        
    def save_datapoint(self, name):
        torch.save([self.feature, self.label], "data/"+name+".pth")
     
def get_test_dataset_LL(task_num, task_name, train_size, first_steps, hps_env, hps_train, hps_model):
    
    if hps_train["current_method"] in ["ours", "BCFLOW_Ddemoonly", "ours_related", "ours_fourwayrelated"]:
        assert train_size % first_steps == 0, "false train size!"
        traj = torch.load("../../../data/fetchreach/"+task_name+".pt")
        
        if task_name.find(".5") != -1: # to limit the dataset size of TS single flow
            train_size = hps_train["transfer_size"]
        
        if train_size > 0: traj = traj[:train_size // first_steps]
        print(train_size, first_steps, len(traj))
        
        feature, label = torch.cat([traj[i]["observations"] for i in range(len(traj))], dim=0).float(), torch.cat([traj[i]["actions"] for i in range(len(traj))], dim=0).float()
        idx = torch.randperm(feature.shape[0])
        feature, label = feature[idx], label[idx]
        p = int(0.8 * feature.shape[0])
        train_feature = feature[:p]
        train_label = label[:p]
        test_feature = feature[p:]
        test_label = label[p:]
        
    elif hps_train["current_method"] in ["BCFLOW_Ddemoonly_withdatabase", "ours_withdatabase"]:
        traj = torch.load("../../../data/fetchreach/"+task_name+".pt") 
        
        if task_name.find(".5") != -1: # to limit the dataset size of TS single flow
            train_size = hps_train["transfer_size"]
        
        if train_size > 0: traj = traj[:train_size // first_steps]
        
        feature, label = [], [] 
        for i in range(len(traj)):
            for j in range(first_steps - 1):
                feature.append(torch.cat([traj[i]["observations"][j].view(1, -1), traj[i]["observations"][j+1].view(1, -1)], dim=1))
                label.append(traj[i]["actions"][j].view(1, -1))
        feature, label = torch.cat(feature, dim=0).float(), torch.cat(label, dim=0).float()
        idx = torch.randperm(feature.shape[0])
        feature, label = feature[idx], label[idx]
        p = int(0.8 * feature.shape[0])
        train_feature = feature[:p]
        train_label = label[:p]
        test_feature = feature[p:]
        test_label = label[p:]
        
         
    elif hps_train["current_method"] in ["BCFLOW_relatedwithDdemo", "BCFLOW_all", "BCFLOW_allwithoutDdemo", "BCFLOW_relatedwithoutDdemo", "BCFLOW_fourwayrelatedwithDdemo"]:
        
        if hps_train["current_method"] == "BCFLOW_relatedwithDdemo":
            if task_name.find("4.5") != -1: lst = ["fetchhard_hidden_4.0", "fetchhard_hidden_5.0", "fetchhard_hidden_4.5"]
            elif task_name.find("5.5") != -1: lst = ["fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_5.5"]
            elif task_name.find("6.5") != -1: lst = ["fetchhard_hidden_6.0", "fetchhard_hidden_7.0", "fetchhard_hidden_6.5"]
            elif task_name.find("7.5") != -1: lst = ["fetchhard_hidden_7.0", "fetchhard_hidden_0.0", "fetchhard_hidden_7.5"]
        elif hps_train["current_method"] == "BCFLOW_fourwayrelatedwithDdemo":
            if task_name.find("4.5") != -1: lst = ["fetchhard_hidden_3.0", "fetchhard_hidden_4.0", "fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_4.5"]
            elif task_name.find("5.5") != -1: lst = ["fetchhard_hidden_4.0", "fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_7.0", "fetchhard_hidden_5.5"]
            elif task_name.find("6.5") != -1: lst = ["fetchhard_hidden_5.0", "fetchhard_hidden_6.0", "fetchhard_hidden_7.0", "fetchhard_hidden_0.0", "fetchhard_hidden_6.5"]
            elif task_name.find("7.5") != -1: lst = ["fetchhard_hidden_6.0", "fetchhard_hidden_7.0", "fetchhard_hidden_0.0", "fetchhard_hidden_1.0", "fetchhard_hidden_7.5"]
        elif hps_train["current_method"] == "BCFLOW_relatedwithoutDdemo":
            if task_name.find("4.5") != -1: lst = ["fetchhard_hidden_4.0", "fetchhard_hidden_5.0"]
            elif task_name.find("5.5") != -1: lst = ["fetchhard_hidden_5.0", "fetchhard_hidden_6.0"]
            elif task_name.find("6.5") != -1: lst = ["fetchhard_hidden_6.0", "fetchhard_hidden_7.0"]
            elif task_name.find("7.5") != -1: lst = ["fetchhard_hidden_7.0", "fetchhard_hidden_0.0"]
        elif hps_train["current_method"] in ["BCFLOW_all", "BCFLOW_allwithoutDdemo"]:
            lst = ["fetchhard_hidden_"+str(i)+".0" for i in range(8)]
            if hps_train["current_method"] == "BCFLOW_all": lst.append(task_name)
        
        all_train_feature, all_train_label, all_test_feature, all_test_label = [], [], [], []
        
        for k, name in enumerate(lst):
            traj = torch.load("../../../data/fetchreach/"+name+".pt") # SB4: metaworld traj: lunarlander
            if hps_train["current_method"] in ["BCFLOW_all", "BCFLOW_relatedwithDdemo", "BCFLOW_fourwayrelatedwithDdemo"] and name.find(".5") != -1:
                train_size = hps_train["transfer_size"]
            if train_size > 0: traj = traj[:train_size // first_steps] # SB4: metaworld traj: lunarlander
        
            feature, label = torch.cat([traj[i]["observations"] for i in range(len(traj))], dim=0).float(), torch.cat([traj[i]["actions"] for i in range(len(traj))], dim=0).float()
             
            idx = torch.randperm(feature.shape[0])
            feature, label = feature[idx], label[idx]
            p = int(0.8 * feature.shape[0])
            train_feature = feature[:p]
            train_label = label[:p]
            test_feature = feature[p:]
            test_label = label[p:]
            
            all_train_feature.append(train_feature)
            all_train_label.append(train_label)
            all_test_feature.append(test_feature)
            all_test_label.append(test_label)
            
        train_feature = torch.cat(all_train_feature, dim=0)
        train_label = torch.cat(all_train_label, dim=0)
        test_feature = torch.cat(all_test_feature, dim=0)
        test_label = torch.cat(all_test_label, dim=0)
    
        # 
    print("shape:", train_feature.shape, test_feature.shape, train_label.shape, test_label.shape)
    # exit(0)
    # average_reward = np.sum(np.concatenate([traj[idx[i]]["rewards"] for i in range(p, len(traj))], axis=0))
    # print("average_reward:", average_reward / len(traj))
    
    train_dataset = Testdataset(train_feature, train_label, task_num)
    test_dataset = Testdataset(test_feature, test_label, task_num)
    if task_num < hps_env["task_num"]: train_loader = DataLoader(train_dataset, batch_size=hps_train["batch_size_train"], shuffle=True, drop_last=True)
    else: train_loader = DataLoader(train_dataset, batch_size=hps_train["batch_size_transfer"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader