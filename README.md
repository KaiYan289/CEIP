# CEIP

This repository is the code of CEIP and PARROT method for NeurIPS 2022 submission "CEIP: Combining Explicit and Implicit Priors for Reinforcement Learning with Demonstrations".

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

For fetchreach experiment, we use a special set of environment with python 3.8.5. See env_fetchreach.txt for a complete table of packages.

We re-write the code of gym robotics environment, which is uploaded in CEIP/robotics. Before running RL for fetchreach, you need to substitute the gym/envs/robotics folder of gym package with our CEIP/robotics folder.

### Others

For kitchen-SKiLD, kitchen-FIST and office, we use another set of environment with python 3.8.5. See env_others.txt for a complete table of packages.

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
change XXXXXXX to your key and username for wandb. See wandb official website https://docs.wandb.ai/ for this.

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



### kitchen-SKiLD

### kitchen-FIST

### office


## Reference

[1] K. Pertsch, Y. Lee, Y. Wu, and J. J. Lim. Demonstration-guided reinforcement learning with learned skills. In CoRL, 2021.

