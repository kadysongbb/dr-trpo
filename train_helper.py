"""
Distributionally Robust Trust Region Policy Optimization
Training Helper Functions

Author: Jun Song (kadysongbb.github.io)
Based on Patrick Coady's implementation (https://github.com/pat-coady/trpo)
"""

import numpy as np
import scipy.signal

def run_episode(env, policy, animate=False):
    """Run single episode with option to animate.

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len)
        actions: shape = (episode len)
        rewards: shape = (episode len)
    """
    obs = env.reset()
    observes, actions, rewards = [], [], []
    done = False
    while not done:
        if animate:
            env.render() 
        observes.append(obs)
        action = policy.sample(obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    return (np.asarray(observes), np.asarray(actions), np.array(rewards, dtype=np.float32))


def run_policy(env, policy, episodes, logger):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
    """
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards = run_episode(env, policy)
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards}
        trajectories.append(trajectory)
    logger.log({'_AvgRewardSum': np.mean([t['rewards'].sum() for t in trajectories])})
    return trajectories

def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma, logger):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        rewards = trajectory['rewards'] 
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew
    logger.log({'_AvgDiscountedRewardSum': np.mean([t['disc_sum_rew'][0] for t in trajectories])})


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values.flatten()

def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        rewards = trajectory['rewards'] 
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def find_disc_freqs(trajectories, sta_num, gamma):
    """ Esimate unnormalized discounted visitation frequencies. 

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
    """
    num_trajectories = len(trajectories)
    disc_freqs = np.zeros(sta_num)
    for trajectory in trajectories:
        observes = trajectory['observes']
        for i in range(len(observes)):
            disc_freqs[observes[i]] += (gamma**i)/num_trajectories
    return disc_freqs

        
def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N,)
        actions: shape = (N,)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    return observes, actions, advantages, disc_sum_rew

def log_batch_stats(observes, actions, advantages, disc_sum_rew, episode, logger):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })