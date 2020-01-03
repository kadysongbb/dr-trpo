"""
Distributionally Robust Trust Region Policy Optimization 
Distributionally Robust Policy Class

Author: Jun Song (kadysongbb.github.io)

Works with "Box" Observation Space, "Discrete" Action Space
DRPolicyCont: Use KL Constraint. 
DRPolicyContWass: Use Wasserstein Constraint. 
"""

import numpy as np
import random

class DRPolicyCont(object):
    def __init__(self, discretize_level, sta_dim, upper_bound, lower_bound, act_num):
        """
        Args:
            discretize_level: level for discretization
            sta_dim: dimension of state space
            upper_bound: upper bound of state space - an array with size "sta_dim"
            lower_bound: lower bound of state space - an array with size "sta_dim"
            act_num: number of actions 
        """
        self.discretize_level = discretize_level
        self.sta_dim = sta_dim
        self.sta_num = discretize_level**sta_dim
        self.act_num = act_num
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        # initial policy PMF π(a|s): a list of 'sta_num' arrays, each array has size 'act_num'
        self.distributions = []
        for i in range(self.sta_num):
            self.distributions.append(np.ones(act_num)/act_num)

    def sample(self, obs):
        """Draw sample from policy."""
        # an array of size 'act_num'
        int_obs = self.discretize(obs)
        distribution = self.distributions[int_obs];
        # sample a cumulative probability
        cdf_sample = random.random();
        # sample the action using the cumulative probability
        cdf = 0
        for i in range(len(distribution)):
            cdf += distribution[i]
            if cdf > cdf_sample:
                return i

    def update(self, observes, actions, advantages):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, numpy array of size N
            actions: actions, numpy array of size N
            advantages: advantages, numpy array of size N
        """
        # reformulate advantages as a list of 'sta_num' arrays, each array has size 'act_num'
        # if advantage(si,aj) is not estimated by GAE, set it to zero
        # if advantage(si,aj) is estimated multiple times by GAE, set it to the average of estimates 
        all_advantages = []
        count = []
        for i in range(self.sta_num):
            all_advantages.append(np.zeros(self.act_num))
            count.append(np.zeros(self.act_num))
        for i in range(len(observes)):
            int_obs = self.discretize(observes[i])
            all_advantages[int_obs][actions[i]] += advantages[i]
            count[int_obs][actions[i]] += 1
        for i in range(self.sta_num):
            for j in range(self.act_num):
                if count[i][j] != 0:
                    all_advantages[i][j] = all_advantages[i][j]/count[i][j]
        
        # compute the new policy
        beta = 1
        old_distributions = self.distributions
        for i in range(self.sta_num):
            denom = np.sum(old_distributions[i]*np.exp(all_advantages[i]/beta))
            self.distributions[i] = np.exp(all_advantages[i]/beta)*old_distributions[i]/denom

    def discretize(self, obs):
        """Convert continuous state to discrete state
        Args:
            obs: an observation, numpy array of size "sta_dim"
        """
        int_obs = 0
        for i in range(self.sta_dim):
            int_obs = int_obs + ((obs[i]-self.lower_bound[i])*self.discretize_level)\
            //(self.upper_bound[i] - self.lower_bound[i])*(self.discretize_level**(self.sta_dim-i-1))
        return int(int_obs)

class DRPolicyContWass(object):
    def __init__(self, discretize_level, sta_dim, upper_bound, lower_bound, act_num):
        """
        Args:
            discretize_level: level for discretization
            sta_dim: dimension of state space
            upper_bound: upper bound of state space - an array with size "sta_dim"
            lower_bound: lower bound of state space - an array with size "sta_dim"
            act_num: number of actions 
        """
        self.discretize_level = discretize_level
        self.sta_dim = sta_dim
        self.sta_num = discretize_level**sta_dim
        self.act_num = act_num
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        # initial policy PMF π(a|s): a list of 'sta_num' arrays, each array has size 'act_num'
        self.distributions = []
        for i in range(self.sta_num):
            self.distributions.append(np.ones(act_num)/act_num)

    def sample(self, obs):
        """Draw sample from policy."""
        # an array of size 'act_num'
        int_obs = self.discretize(obs)
        distribution = self.distributions[int_obs];
        # sample a cumulative probability
        cdf_sample = random.random();
        # sample the action using the cumulative probability
        cdf = 0
        for i in range(len(distribution)):
            cdf += distribution[i]
            if cdf > cdf_sample:
                return i

    def update(self, observes, actions, advantages):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, numpy array of size N
            actions: actions, numpy array of size N
            advantages: advantages, numpy array of size N
        """
        # reformulate advantages as a list of 'sta_num' arrays, each array has size 'act_num'
        # if advantage(si,aj) is not estimated by GAE, set it to zero
        # if advantage(si,aj) is estimated multiple times by GAE, set it to the average of estimates 
        all_advantages = []
        count = []
        for i in range(self.sta_num):
            all_advantages.append(np.zeros(self.act_num))
            count.append(np.zeros(self.act_num))
        for i in range(len(observes)):
            int_obs = self.discretize(observes[i])
            all_advantages[int_obs][actions[i]] += advantages[i]
            count[int_obs][actions[i]] += 1
        for i in range(self.sta_num):
            for j in range(self.act_num):
                if count[i][j] != 0:
                    all_advantages[i][j] = all_advantages[i][j]/count[i][j]
        
        # compute Qijk
        beta = 10
        best_k = [[0] * self.act_num for i in range(self.sta_num)]
        for i in range(self.sta_num):
            for j in range(self.act_num):
                opt_k = 0
                opt_val = all_advantages[i][opt_k] - beta*self.calc_d(opt_k,j)
                for k in range(self.act_num):
                    cur_val = all_advantages[i][k] - beta*self.calc_d(k,j)
                    if cur_val > opt_val:
                        opt_k = k
                        opt_val = cur_val
                best_k[i][j] = opt_k

        # compute the new policy 
        old_distributions = self.distributions
        self.distributions = []
        for i in range(self.sta_num):
            self.distributions.append(np.zeros(self.act_num))
        for i in range(self.sta_num):
            for k in range(self.act_num):
                for j in range(self.act_num):
                    if k == best_k[i][j]:
                        self.distributions[i][k] += old_distributions[i][j]


    def calc_d(self, aj, ak):
        if aj == ak:
            return 0
        else:
            return 1

    def discretize(self, obs):
        """Convert continuous state to discrete state
        Args:
            obs: an observation, numpy array of size "sta_dim"
        """
        int_obs = 0
        for i in range(self.sta_dim):
            int_obs = int_obs + ((obs[i]-self.lower_bound[i])*self.discretize_level)\
            //(self.upper_bound[i] - self.lower_bound[i])*(self.discretize_level**(self.sta_dim-i-1))
        return int(int_obs)