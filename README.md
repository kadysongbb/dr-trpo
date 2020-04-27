## Optimistic Dristributionally Robust Policy Optimization (ODRPO)

### Tabular: 
* Two algorithms implemented: ODRPO KL, ODRPO Wasserstein
* This version can run on OpenAI Gym toy text environment, i.e. Taxi-v3, Nchain-v0.
* Policy is not parametrized (i.e. no Neural Network for policies). It's represented as a list that contains PMF of π(·|s) for all states. 

![Performance Graph 1](tabular.png?raw=true)

### Discrete Control: 
* Three algorithms implemented: ODRPO KL, ODRPO Wasserstein, A2C (baseline)
* This version supports OpenAI Gym environments with discrete action space. 
* Policy is parameterized as a neural net that maps from state s to the PMF of π(·|s) 

![Performance Graph 2](discrete.png?raw=true)

### Continuous Control: 
