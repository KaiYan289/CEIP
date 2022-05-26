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

## Dependency

As multiple sets of environment is needed, virtual environment like anaconda are strongly recommended. 

### Fetchreach

For fetchreach experiment, we use a special set of environment with python 3.8.5. See env_fetchreach.txt for a complete table of packages.



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

## Detailed Commands for Each Result


### Fetchreach



## Reference

[1] K. Pertsch, Y. Lee, Y. Wu, and J. J. Lim. Demonstration-guided reinforcement learning with learned skills. In CoRL, 2021.

