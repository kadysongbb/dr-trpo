"""
Distributionally Robust Trust Region Policy Optimization 
Distributionally Robust Policy Class

Author: Jun Song (kadysongbb.github.io)

Works with "Discrete" Observation Space, "Discrete" Action Space
DRPolicyKL: Use KL Constraint. 
DRPolicyWass: Use Wasserstein Constraint. 
"""

import numpy as np
import random

class DRPolicyKL(object):
    def __init__(self, sta_num, act_num):
        """
        Args:
            sta_num: number of states
            act_num: number of actions
        """
        # initial policy PMF π(a|s): a list of 'sta_num' arrays, each array has size 'act_num'
        # For KL constraint, PMF should not have zero 
        self.sta_num = sta_num
        self.act_num = act_num
        self.distributions = []
        for i in range(sta_num):
            self.distributions.append(np.ones(act_num)/act_num)

    def sample(self, obs):
        """Draw sample from policy."""
        # an array of size 'act_num'
        distribution = self.distributions[obs];
        # sample a cumulative probability
        cdf_sample = random.random();
        # sample the action using the cumulative probability
        cdf = 0
        for i in range(len(distribution)):
            cdf += distribution[i]
            if cdf >= cdf_sample:
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
           all_advantages[observes[i]][actions[i]] += advantages[i]
           count[observes[i]][actions[i]] += 1
        for i in range(self.sta_num):
            for j in range(self.act_num):
                if count[i][j] != 0:
                    all_advantages[i][j] = all_advantages[i][j]/count[i][j]

        # compute the new policy
        beta = 1
        old_distributions = self.distributions
        for i in range(self.sta_num):
            denom = np.sum(np.exp(all_advantages[i]/beta)*old_distributions[i])
            self.distributions[i] = np.exp(all_advantages[i]/beta)*old_distributions[i]/denom


class DRPolicyWass(object):
    def __init__(self, sta_num, act_num):
        """
        Args:
            sta_num: number of states
            act_num: number of actions
        """
        # initial policy PMF π(a|s): a list of 'sta_num' arrays, each array has size 'act_num'
        # For KL constraint, PMF should not have zero 
        self.sta_num = sta_num
        self.act_num = act_num
        self.distributions = []
        for i in range(sta_num):
            self.distributions.append(np.ones(act_num)/act_num)

    def sample(self, obs):
        """Draw sample from policy."""
        # an array of size 'act_num'
        distribution = self.distributions[obs];
        # sample a cumulative probability
        cdf_sample = random.random();
        # sample the action using the cumulative probability
        cdf = 0
        for i in range(len(distribution)):
            cdf += distribution[i]
            if cdf >= cdf_sample:
                return i

    def update(self, observes, actions, advantages, disc_freqs):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, numpy array of size N
            actions: actions, numpy array of size N
            advantages: advantages, numpy array of size N
            disc_freqs: discounted visitation frequencies, numpy array of size 'sta_num'
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
           all_advantages[observes[i]][actions[i]] += advantages[i]
           count[observes[i]][actions[i]] += 1
        for i in range(self.sta_num):
            for j in range(self.act_num):
                if count[i][j] != 0:
                    all_advantages[i][j] = all_advantages[i][j]/count[i][j]

        # compute Qijk
        opt_beta = self.find_opt_beta(0.01, 0.01, all_advantages, disc_freqs, 0.01)
        best_k = self.find_best_k(opt_beta, all_advantages)

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
        """Calculate the distance between two actions."""
        if ak == aj:
            return 0
        else:
            return 1

    def find_best_k(self, beta, all_advantages):
        """Find argmax_k {A(si, ak) - β*d(ak,aj)}."""
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
        return best_k

    def find_opt_beta(self, delta, init_beta, all_advantages, disc_freqs, precision):
        """Find optimal beta using gradient descent."""
        cur_beta = init_beta
        next_beta = init_beta + precision + 1e-3
        while abs(next_beta - cur_beta) > precision:
            cur_beta = next_beta
            best_k = self.find_best_k(cur_beta, all_advantages)
            gradient = delta 
            for i in range(self.sta_num):
                for j in range(self.act_num):
                    gradient += -disc_freqs[i]*self.distributions[i][j]*self.calc_d(best_k[i][j], j)
            next_beta = cur_beta - 0.1*gradient
        return next_beta
        