## Dristributionally Robust Policy Optimization 

### Tabular: 
* Two algorithms implemented: DR TRPO KL, DR TRPO Wasserstein
* This version can run on OpenAI Gym toy text environment, i.e. Taxi-v3, Nchain-v0.
* Policy is not parametrized (i.e. no Neural Network for policies). It's represented as a list that contains PMF of π(a|s) for all states. 

### Classic Control: 
* Three algorithms implemented: DR TRPO KL, DR TRPO Wasserstein, A2C (baseline)
* This version can run on OpenAI Gym classic control environment. 
* Policy is parameterized as a neural net that maps from state s to the PMF of π(a|s) when action space is discrete.
