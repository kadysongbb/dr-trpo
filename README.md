# Optimistic Dristributionally Robust Policy Optimization (ODRPO)

## Requirements: 
* numpy 
* mujuco - see: https://github.com/openai/mujoco-py
* tensorflow
* pytorch

## How to Run: 
Run train_drpo*.ipynb to train ODRPO (or python main.py --mode "odrpo" for continuous control) <br />
Run view_training*.ipynb to see the training results

## Description and Performance: 
### Tabular: 
* Two algorithms implemented: ODRPO KL, ODRPO Wasserstein
* Supports OpenAI Gym environments with discrete observation and discrete action space, i.e. Taxi-v3, Nchain-v0
* Policy is not parametrized (i.e. no Neural Network for policies). It's represented as a list that contains PMFs of π(·|s) for all states 

![Performance Graph 1](tabular.png?raw=true)

### Locomotion Tasks:
#### Discrete Control: 
* Three algorithms implemented: ODRPO KL, ODRPO Wasserstein, A2C (baseline)
* Supports OpenAI Gym environments with continuous observation and discrete action space, i.e. CartPole-v1, Acrobot-v1
* Policy is parameterized as a neural net that maps from state s to the PMF of π(·|s) 

#### Continuous Control: 
* One algorithm implemented: ODRPO KL (based on [GAC](https://github.com/gwbcho/dpo-replication))
* Supports OpenAI Gym environments with continuous observation and continous action space

![Performance Graph 2](locomotion.png?raw=true)
