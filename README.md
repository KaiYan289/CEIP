# CEIP

This repository is the code of CEIP and PARROT method for NeurIPS 2022 proceedings "CEIP: Combining Explicit and Implicit Priors for Reinforcement Learning with Demonstrations". The project page is https://289371298.github.io/jekyll/update/2022/10/25/CEIP.html.

The dataset can be downloaded at https://www.dropbox.com/s/7wg61g7qjmeg0bv/data.zip?dl=0 . The "data" folder is already structured and should be placed under the CEIP directory.

## File Structure

CEIP/data: the data for the experiments, which are downloaded through the link above.

CEIP/dataset: the dataloaders for each experiment.

CEIP/demo: the entry of flow training. It contains two subfolders, CEIP/ours and CEIP/PARROT; each of the two subfolders contains four sub-subfolders, one for each experiment (fetchreach, kitchen_FIST, kitchen_SKiLD, office).

CEIP/hyperparams: the hyperparameters for each experiment.

CEIP/log: the logged results for RL runs.

CEIP/RL: the code for RL runs. 

CEIP/robotics: substituting part of the code in gym for fetchreach; see dependency section below for details.

## Dependency

As multiple sets of environment is needed, virtual environment like anaconda are strongly recommended. 

### Fetchreach

For fetchreach experiment, we use a special set of environment with python 3.8.5. The requirements are:
```
torch==1.10.0
gym==0.21.0
numpy==1.21.4
h5py==2.10.0
stable-baselines3==1.3.0
imageio
wandb
tqdm
matplotlib
```
We re-write the code of gym robotics environment, which is uploaded in CEIP/robotics. Before running RL for fetchreach, you need to substitute the gym/envs/robotics folder of gym package with our CEIP/robotics folder.

### Others

For kitchen-SKiLD, kitchen-FIST and office, we use another set of environment with python 3.8.5. The requirements are:
```
mujoco-py==2.0.2.13
stable-baselines3==1.3.0
torch==1.10.0
gym>=0.15.4
h5py==2.10.0
imageio
wandb
tqdm
matplotlib
```
For kitchen environment, a fork of **d4rl** by the author of SKiLD[1] is required. The URL for the repo is https://github.com/kpertsch/d4rl .

For office environment, a fork of **roboverse** by the author of SKiLD is required The URL for the repo is https://github.com/VentusYue/roboverse .

**Note that mujoco200 is required for these environments. Also, do not install the original version of d4rl and roboverse for substitution.**

## Running Code

1. Clone the repository.

2. Install the dependencies as stated in the dependency section, and download the dataset.

3. In the CEIP folder, type the following command: 
``` 
export PYTHONPATH='./':$PYTHONPATH 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:(path to mujoco)/.mujoco/mujoco200/bin
```

4. go into the sub-subfolder for demos (for example: demo/ours/fetchreach), type
```
mkdir code_backup
mkdir models
```

5. open the demo file to run; in the line 
```
wandb.login(key=XXXXXXX)
wandb.init(entity=XXXXXXX, project= ...
```
change XXXXXXX to your key and username for wandb. See wandb official website https://docs.wandb.ai/ for this. We use XXXXXXX for anonimity.

6. run the code for training flow (see the next section for detailed commands).

7. get back to the main CEIP directory; run corresponding RL code (for example: ```python RL/RL_fetch.py --seed 19 --modelseed 1000009 --direction 4.5 --method ours```)

8. The results of RL are stored in CEIP/log directory; log into your wandb account to check flow training results.

## Detailed Steps for Reproducing Results


### Fetchreach

#### Ours
Training flows (in demo/ours/fetchreach):

**ours_noTS_noEX**: set line 50 of demo_ours.py to "setenv(args.env_name, 10, 4, **8**)", then run ```python demo_ours.py --seed 1000009```.

**ours_TS_noEX**: set line 50 of demo_ours.py to "setenv(args.env_name, 10, 4, **9**)", then run ```python demo_ours.py --seed 1000009```.

**ours_noTS_EX**: set line 50 of demo_ours_withdatabase.py to "setenv(args.env_name, 10, 4, **8**)", then run ```python demo_ours_withdatabase.py --seed 1000009```.

**ours_TS_EX**: set line 50 of demo_ours_withdatabase.py to "setenv(args.env_name, 10, 4, **9**)", then run ```python demo_ours_withdatabase.py --seed 1000009```.

**ours_4way** (below are part of our ablation test; see paper for their meaning): ```python demo_ours_fourwayrelated.py --seed 1000009```

**ours_2way**: ```python demo_ours_related.py --seed 1000009```

run RL (in main directory):

**ours_noTS_noEX**: ```python RL/RL_fetch.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method ours``` (change 4.5 to 5.5, 6.5, 7.5 for other directions)

**ours_TS_noEX**: ```python RL/RL_fetch.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method ours_withTS```

**ours_noTS_EX**: ```python RL/RL_fetch_withdatabase.py --seed 19 --modelseed 1000009 --direction 4.5 --transize 1600 --transfersize 160 --method ours_withdatabase --pushforward no```

**ours_TS_EX**: ```python RL/RL_fetch_withdatabase.py --seed 19 --modelseed 1000009 --direction 4.5 --transize 1600 --transfersize 160 --method ours_withTS_withdatabase --pushforward no```

**ours_noTS_EX_forward**: ```python RL/RL_fetch_withdatabase.py --seed 19 --modelseed 1000009 --direction 4.5 --transize 1600 --transfersize 160 --method ours_withdatabase --pushforward yes```

**ours_TS_EX_forward**: ```python RL/RL_fetch_withdatabase.py --seed 19 --modelseed 1000009 --direction 4.5 --transize 1600 --transfersize 160 --method ours_withTS_withdatabase --pushforward yes```

**ours_4way**: ```python RL/RL_fetch.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method ours_fourwayrelated```

**ours_2way**: ```python RL/RL_fetch.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method ours_related```

#### PARROT

Training flows (in demo/PARROT/fetchreach): 

**PARROT-TATS**: ```python demo_BCFlow_all.py --seed 1000009```

**PARROT-TA**: ```python demo_BCFLOW_allwithoutDdemo.py --seed 1000009``` 

**PARROT_4way+TS**: ```python demo_BCFLOW_fourwayrelatedwithDdemo.py --seed 1000009```

**PARROT_2way**: ```python demo_BCFLOW_relatedwithoutDdemo.py --seed 1000009```

**PARROT_2way+TS**: ```python demo_BCFLOW_relatedwithDdemo.py --seed 1000009```

**PARROT_TS_noEX**: ```python demo_Ddemoonly.py --seed 1000009```

**PARROT_TS_EX**: ```python demo_Ddemoonly_withdatabase.py --seed 1000009```

run RL (in the main directory):

**PARROT-TATS**: ```python RL/RL_fetch.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method alone_all```

**PARROT-TA**: ```python RL/RL_fetch.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method alone_allwithoutDdemo``` 

**PARROT_4way+TS**: ```python RL/RL_fetch.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method alone_fourwayrelatedwithDdemo```

**PARROT_2way**: ```python RL/RL_fetch.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method alone_relatedwithoutDdemo```

**PARROT_2way+TS**: ```python RL/RL_fetch.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method alone_relatedwithDdemo```

**PARROT_TS_noEX**: ```python RL/RL_fetch.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method alone_Ddemoonly```

**PARROT_TS_EX**: ```python RL/RL_fetch_withdatabase.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method alone_withdatabase --pushforward no```

**PARROT_TS_EX_forward**: ```python RL/RL_fetch_withdatabase.py --seed 19 --modelseed 1000009 --direction 4.5 --trainsize 1600 --transfersize 160 --method alone_withdatabase --pushforward yes```

### kitchen-SKiLD

#### ours

Training flows (in demo/ours/kitchen_SKiLD):

**ours-noTS-noEX**: set line 70 of demo_ours_forall.py to "setenv(args.env_name, 60, 9, **24**)"; set line 60 of demo_ours_forall.py to "hps_model_setter("type", **"1layer_single"**)"; run ```python demo_ours_forall.py --seed 1000009```

**ours-TS-noEX**: set line 70 of demo_ours_forall.py to "setenv(args.env_name, 60, 9, **25**)"; set line 60 of demo_ours_forall.py to "hps_model_setter("type", **"1layer_debug"**)"; run ```python demo_ours_forall.py --seed 1000009```

**ours-noTS-EX**: set line 70 of demo_ours_forall.py to "setenv(args.env_name, 60, 9, **24**)"; set line 60 of demo_ours_forall.py to "hps_model_setter("type", **"1layer_single"**)"; run ```python demo_ours_forall.py --seed 1000009```

**ours-TS-EX**: set line 70 of demo_ours_forall.py to "setenv(args.env_name, 60, 9, **25**)"; set line 60 of demo_ours_forall.py to "hps_model_setter("type", **"1layer_debug"**)"; run ```python demo_ours_forall.py --seed 1000009```

Run RL (in main directory):

**ours-noTS-noEX**: ```python RL/RL_PPO_formal_nodatabase_kitchen_SKiLD.py --seed 19 --modelseed 1000009 --TL 300 --pushforward no``` (300 for SKiLD-A, 301 for SKiLD-B)


**ours-TS-noEX**: ```python RL/RL_PPO_formal_nodatabase_kitchen_SKiLD.py --seed 19 --modelseed 1000009 --TL 400 --pushforward no``` (400 for SKiLD-A, 401 for SKiLD-B)

**ours-noTS-EX**: ```python RL/RL_PPO_formal_kitchen_SKiLD.py --seed 19 --modelseed 1000009 --TL 100 -- pushforward no``` (100 for SKiLD-A, 101 for SKiLD-B)

**ours-TS-EX**: ```python RL/RL_PPO_formal_kitchen_SKiLD.py --seed 19 --modelseed 1000009 --TL 200 -- pushforward no``` (200 for SKiLD-A, 201 for SKiLD-B)

**ours-noTS-EX-forward**: ```python RL/RL_PPO_formal_kitchen_SKiLD.py --seed 19 --modelseed 1000009 --TL 100 --pushforward yes```

**ours-TS-EX-forward**: ```python RL/RL_PPO_formal_kitchen_SKiLD.py --seed 19 --modelseed 1000009 --TL 200 -- pushforward yes```

#### PARROT

Training flows (in demo/PARROT/kitchen_SKiLD):

**PARROT-TA**, **PARROT-TS-noEX**: ```python demo_ours_forall_nodatabase.py --seed 1000009```

**PARROT-TS-EX**: ```python demo_ours_forall_withdatabase.py --seed 1000009```

Run RL (in main directory):

**PARROT-TA**: ```python RL/RL_PPO_formal_nodatabase_kitchen_SKiLD.py --seed 19 --modelseed 1000009 --TL 600``` (600 for SKiLD-A, 601 for SKiLD-B)

**PARROT-TS-noEX**: ```python RL/RL_PPO_formal_nodatabase_kitchen_SKiLD.py --seed 19 --modelseed 1000009 --TL 500``` (500 for SKiLD-A, 501 for SKiLD-B)

**PARROT-TS-EX**: ```python RL/RL_PPO_formal_kitchen_SKiLD.py --seed 19 --modelseed 1000009 --TL 230 --pushforward no``` (230 for SKiLD-A, 231 for SKiLD-B)

**PARROT-TS-EX-forward**: ```python RL/RL_PPO_formal_kitchen_SKiLD.py --seed 19 --modelseed 1000009 --TL 500 --pushforward yes```

### Others

Kitchen-FIST and office can be run in a similar way to kitchen-SKiLD; see the code for detailed command.


## Reference

[1] K. Pertsch, Y. Lee, Y. Wu, and J. J. Lim. Demonstration-guided reinforcement learning with learned skills. In CoRL, 2021.

