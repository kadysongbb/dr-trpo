"""
Distributionally Robust Trust Region Policy Optimization 
Distributionally Robust Policy Class

Author: Jun Song (kadysongbb.github.io)

Works with "Discrete" Observation Space, "Discrete" Action Space
DRPolicyKL: Use KL Constraint. 
DRPolicyWass: Use Wasserstein Constraint. 
"""

import numpy as np
from scipy import optimize
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
        self.delta = 0.01
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

        def gradient(beta):
            gradient = self.delta
            for s in range(self.sta_num):
                gradient += disc_freqs[s]*np.log(np.sum(np.exp(all_advantages[s]/beta)*self.distributions[s]))
                numerator = np.sum(np.exp(all_advantages[s]/beta)*all_advantages[s]*self.distributions[s])
                denom = beta*np.sum(np.exp(all_advantages[s]/beta)*self.distributions[s])
                gradient -= disc_freqs[s]*numerator/denom
            return gradient

        def objective(beta):
            objective = beta*self.delta
            for s in range(self.sta_num):
                objective += beta*disc_freqs[s]*np.log(np.sum(np.exp(all_advantages[s]/beta)*self.distributions[s]))
            return objective


        minimizer_kwargs = {"method": "BFGS", "jac": gradient}
        beta = optimize.basinhopping(objective, 0.5, minimizer_kwargs=minimizer_kwargs,
              niter=20)
        beta = beta.x[0]
        print(beta)

        # compute the new policy
        old_distributions = self.distributions
        for s in range(self.sta_num):
            denom = np.sum(np.exp(all_advantages[s]/beta)*old_distributions[s])
            self.distributions[s] = np.exp(all_advantages[s]/beta)*old_distributions[s]/denom


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
        self.delta = 0.01
            
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


        def find_best_j(beta):
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

        def objective(beta):
            objective = beta*self.delta
            best_j  =  find_best_j(beta)
            for s in range(self.sta_num):
                for i in range(self.act_num):
                    opt_j = best_j[s][i]
                    objective += disc_freqs[s]*self.distributions[s][i]*(all_advantages[s][opt_j] - beta*self.calc_d(opt_j, i))
            return  objective

        # if 'Taxi' in env_name:
        #     opt_beta = 2 + 0.8*(np.random.random() - 0.5)
        # if 'Chain' in env_name:
        #     opt_beta = 0.5
        # if 'Cliff' in env_name:
        #     opt_beta = 0.5
        # best_j = self.find_best_j(opt_beta, all_advantages)

        rranges = [(0,4)]
        beta = optimize.dual_annealing(objective, rranges, maxiter = 20)
        beta = beta.x[0]
        print(beta)

        # Q
        best_j = find_best_j(beta)
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
        