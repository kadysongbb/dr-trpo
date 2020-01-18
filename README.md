## Dristributionally Robust Trust Region Policy Optimization 

### Toy Version: 
* Modified based on https://github.com/pat-coady/trpo
* Value is trained on discounted reward sum
* Advantage is estimated with GAE
* Policy is not parametrized (i.e. no Neural Network for policies). This version can run on OpenAI Gym toy text environment, i.e. Taxi-v3, Nchain-v0. 
