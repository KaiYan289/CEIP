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
     
def get_test_dataset_LL(task_num, task_name, train_size, first_steps, hps_env, hps_train,  hps_model):
    traj = torch.load("../../../data/kitchen_FIST/"+"24task-"+hps_env["alphabet"]+"/"+task_name+".pt") # SB4: metaworld traj: lunarlander
    
    if hps_model["type"] == "1layer_debug": # with database
        obss, actions = [], []
        for t in traj:
            # print(t["observations"])
            for j in range(len(t["observations"]) - 1):
                obss.append(np.concatenate([t["observations"][j], t["observations"][j+1]], axis=0).reshape(1, -1))
                actions.append(t["actions"][j].reshape(1, -1))
        
        traj = [{"observations": np.concatenate(obss, axis=0), "actions": np.concatenate(actions, axis=0)}]
    elif hps_model["type"] == "1layer_single":
        traj[0]["observations"] = np.concatenate([traj[i]["observations"] for i in range(len(traj))], axis=0)
        traj[0]["actions"] = np.concatenate([traj[i]["actions"] for i in range(len(traj))], axis=0)
    print(task_num, traj[0]["observations"].shape, traj[0]["actions"].shape)
    # exit(0)
    idx = torch.randperm(traj[0]["observations"].shape[0])
    
    p = int(0.8 * len(idx))
    print(len(traj), traj[0]["observations"].shape[0], traj[0]["actions"].shape[0])
    train_feature, test_feature = torch.cat([torch.from_numpy(traj[0]["observations"][idx[i]]).unsqueeze(0) for i in range(p)], dim=0).float(), torch.cat([torch.from_numpy(traj[0]["observations"][idx[i]]).unsqueeze(0) for i in range(p, traj[0]["observations"].shape[0])], dim=0).float()
    train_label, test_label = torch.cat([torch.from_numpy(traj[0]["actions"][idx[i]]).unsqueeze(0) for i in range(p)], dim=0).float(), torch.cat([torch.from_numpy(traj[0]["actions"][idx[i]]).unsqueeze(0) for i in range(p, traj[0]["observations"].shape[0])], dim=0).float()
    
    
    print(train_feature.shape, test_feature.shape, train_label.shape, test_label.shape)

    if train_size > 0: train_feature, train_label = train_feature[:train_size], train_label[:train_size]
    train_dataset = Testdataset(train_feature, train_label, task_num)
    test_dataset = Testdataset(test_feature, test_label, task_num)
    if task_num < hps_env["task_num"]: train_loader = DataLoader(train_dataset, batch_size=hps_train["batch_size_train"], shuffle=True, drop_last=True)
    else: train_loader = DataLoader(train_dataset, batch_size=hps_train["batch_size_transfer"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    
    return train_loader, test_loader
