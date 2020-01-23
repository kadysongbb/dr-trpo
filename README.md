## Dristributionally Robust Trust Region Policy Optimization 

### Toy Version: 
* Modified based on https://github.com/pat-coady/trpo
* Value is trained on discounted reward sum
* Advantage is estimated with GAE
* Policy is not parametrized (i.e. no Neural Network for policies). This version can run on OpenAI Gym toy text environment, i.e. Taxi-v3, Nchain-v0. 

### Discrete Action:
* For experiments with small-size discrete action space (Classical Control),  we construct the policy neural net that maps from state s to the PMF of π(a|s). For experiments with large-size discrete action space (Atari), we factorize the action space as 3 x 3 x 2 and construct the policy neural net to map from state s to the three PMFs of π(a|s). 
