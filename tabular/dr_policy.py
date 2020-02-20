"""
Distributionally Robust Trust Region Policy Optimization 
Distributionally Robust Policy Class

Author: Jun Song (kadysongbb.github.io)

Works with "Discrete" Observation Space, "Discrete" Action Space
DRPolicyKL: Use KL Constraint. 
DRPolicyWass: Use Wasserstein Constraint. 
"""

import numpy as np
from sklearn.linear_model import LinearRegression

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
        # sample an action
        action = np.random.choice(self.act_num, 1, p=distribution)
        return action[0]

    def update(self, observes, actions, advantages, disc_freqs):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, numpy array of size N
            actions: actions, numpy array of size N
            advantages: advantages, numpy array of size N
        """
        all_advantages = []
        count = []
        x = []
        for i in range(self.sta_num):
            all_advantages.append(np.zeros(self.act_num))
            count.append(np.zeros(self.act_num))
        for i in range(len(observes)):
           all_advantages[observes[i]][actions[i]] += advantages[i]
           count[observes[i]][actions[i]] += 1
        for s in range(self.sta_num):
            for i in range(self.act_num):
                if count[s][i] != 0:
                    all_advantages[s][i] = all_advantages[s][i]/count[s][i]

        # compute the new policy
        # beta = self.find_opt_beta(0.1, all_advantages, disc_freqs, 0.01, 0.1, 1e-2, 1000)
        beta = 1
        old_distributions = self.distributions
        for s in range(self.sta_num):
            denom = np.sum(np.exp(all_advantages[s]/beta)*old_distributions[s])
            self.distributions[s] = np.exp(all_advantages[s]/beta)*old_distributions[s]/denom

    def find_opt_beta(self, init_beta, all_advantages, disc_freqs, delta, gamma, precision, max_iter):
        """Find optimal beta using gradient descent."""
        cur_beta = init_beta
        next_beta = init_beta + precision + 1e-3
        for i in range(max_iter):
            if abs(next_beta - cur_beta) <= precision:
                break
            cur_beta = next_beta
            gradient = delta
            for s in range(self.sta_num):
                gradient += disc_freqs[s]*np.log(np.sum(np.exp(all_advantages[s]/cur_beta)*self.distributions[s]))
                numerator = np.sum(np.exp(all_advantages[s]/cur_beta)*all_advantages[s]*self.distributions[s])
                denom = cur_beta*np.sum(np.exp(all_advantages[s]/cur_beta)*self.distributions[s])
                gradient -= disc_freqs[s]*numerator/denom
            next_beta = cur_beta - gamma*gradient
        return next_beta

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
        # sample an action
        action = np.random.choice(self.act_num, 1, p=distribution)
        return action[0]

    def update(self, observes, actions, advantages, disc_freqs, env_name):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, numpy array of size N
            actions: actions, numpy array of size N
            advantages: advantages, numpy array of size N
            disc_freqs: discounted visitation frequencies, numpy array of size 'sta_num'
            env_name: name of the environment
        """
        all_advantages = []
        count = []
        x = []
        for i in range(self.sta_num):
            all_advantages.append(np.zeros(self.act_num))
            count.append(np.zeros(self.act_num))
        for i in range(len(observes)):
           all_advantages[observes[i]][actions[i]] += advantages[i]
           count[observes[i]][actions[i]] += 1
        for s in range(self.sta_num):
            for i in range(self.act_num):
                if count[s][i] != 0:
                    all_advantages[s][i] = all_advantages[s][i]/count[s][i]

        # compute Q
        # opt_beta = self.find_opt_beta(0.5, all_advantages, disc_freqs, 0.01, 0.1, 1e-2, 1000)
        # opt_beta = self.find_opt_beta2(all_advantages)
        if 'Taxi' in env_name:
            opt_beta = 2 + 0.8*(np.random.random() - 0.5)
        if 'Chain' in env_name:
            opt_beta = 0.5
        if 'Cliff' in env_name:
            opt_beta = 0.5
        best_j = self.find_best_j(opt_beta, all_advantages)

        # compute the new policy 
        old_distributions = self.distributions
        self.distributions = []
        for i in range(self.sta_num):
            self.distributions.append(np.zeros(self.act_num))
        for s in range(self.sta_num):
            for j in range(self.act_num):
                for i in range(self.act_num):
                    if j == best_j[s][i]:
                        self.distributions[s][j] += old_distributions[s][i]

    def calc_d(self, ai, aj):
        """Calculate the distance between two actions."""
        if ai == aj:
            return 0
        else:
            return 1

    def find_best_j(self, beta, all_advantages):
        """Find argmax_j {A(s,aj) - β*d(aj,ai)}."""
        best_j = [[0] * self.act_num for i in range(self.sta_num)]
        for s in range(self.sta_num):
            for i in range(self.act_num):
                opt_j = 0
                opt_val = all_advantages[s][opt_j] - beta*self.calc_d(opt_j,i)
                for j in range(self.act_num):
                    cur_val = all_advantages[s][j] - beta*self.calc_d(j,i)
                    if cur_val > opt_val:
                        opt_j = j
                        opt_val = cur_val
                best_j[s][i] = opt_j
        return best_j

    def find_opt_beta(self, init_beta, all_advantages, disc_freqs, delta, gamma, precision, max_iter):
        """Find optimal beta using gradient descent."""
        cur_beta = init_beta
        next_beta = init_beta + precision + 1e-3
        for i in range(max_iter):
            if abs(next_beta - cur_beta) <= precision:
                break
            cur_beta = next_beta
            best_j = self.find_best_j(cur_beta, all_advantages)
            gradient = delta 
            for s in range(self.sta_num):
                for i in range(self.act_num):
                    gradient += -disc_freqs[s]*self.distributions[s][i]*self.calc_d(best_j[s][i], i)
            next_beta = cur_beta - gamma*gradient
        return next_beta
        
    def find_opt_beta2(self, all_advantages):
        """Assign optimal beta to be the mean of the advantage difference"""
        total_advantage_diff = 0
        for i in range(self.sta_num):
            total_advantage_diff += np.max(all_advantages[i]) - np.min(all_advantages[i])
        beta = total_advantage_diff/self.sta_num
        return beta