## Optimistic Dristributionally Robust Policy Optimization (ODRPO)


### How to Run: 


### Requirements: 
* numpy 
* mujuco - see: https://github.com/openai/mujoco-py
* tensorflow
* pytorch

### Description and Performance: 
#### Tabular: 
* Two algorithms implemented: ODRPO KL, ODRPO Wasserstein
* Supports OpenAI Gym environment with discrete observation and discrete action space, i.e. Taxi-v3, Nchain-v0
* Policy is not parametrized (i.e. no Neural Network for policies). It's represented as a list that contains PMF of π(·|s) for all states 

![Performance Graph 1](tabular.png?raw=true)

#### Locomotion Tasks:
##### Discrete Control: 
* Three algorithms implemented: ODRPO KL, ODRPO Wasserstein, A2C (baseline)
* Supports OpenAI Gym environments with continuous observation and discrete action space, i.e. CartPole-v1, Acrobot-v1
* Policy is parameterized as a neural net that maps from state s to the PMF of π(·|s) 

##### Continuous Control: 
* Supports OpenAI Gym environments with continuous observation and continous action space
* GAC network with ODRPO KL update function 

![Performance Graph 2](locomotion.png?raw=true)
