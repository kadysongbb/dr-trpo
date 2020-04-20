import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from collections import namedtuple


class ActionSampler:
    """
    Sampling actions from a given actor by feeding samples from a uniform distribution into the
    actor network.
    """

    def __init__(self, action_dim):
        self.dim = action_dim

    def get_actions(self, actor, states, actions=None):
        """
        Actions are obtained from the actor network.
        """
        if states.shape.rank > 1:
            batch_size = states.shape[0]
        else:
            batch_size = 1
        return actor(
            states,
            tf.random.uniform((batch_size, self.dim), minval=0.0, maxval=1.0),
            actions
        )


class ReplayBuffer:
    '''
    A simple FIFO experience replay buffer.
    Adapted from
    https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py (with The MIT License)
    keep the shape in __init__ in mind.
    '''

    def __init__(self, obs_dim, act_dim, size):
        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it'])
        # (this_state, this_action, this_reward, next_state, this_is_terminal)

        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.size_list = range(self.size)

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        prev_size = self.size
        self.size = min(self.size+1, self.max_size)
        if prev_size < self.size:
            self.size_list = range(self.size)

    def sample_batch(self, batch_size=32):
        idxs = random.sample(self.size_list, batch_size)
        self.transitions.s = tf.convert_to_tensor(self.obs1_buf[idxs])
        self.transitions.a = tf.convert_to_tensor(self.acts_buf[idxs])
        self.transitions.r = tf.convert_to_tensor(self.rews_buf[idxs])
        self.transitions.sp = tf.convert_to_tensor(self.obs2_buf[idxs])
        self.transitions.it = tf.convert_to_tensor(self.done_buf[idxs])
        return self.transitions


def update(target, source, rate):
    """
    update function.
    when tau = 1, then it's just assignment, i.e. hard update
    Args:
        target (tf.Module): target model
        source (tf.Module): source model
    """
    target_params = target.trainable_variables
    source_params = source.trainable_variables
    for t, s in zip(target_params, source_params):
        t.assign(t * (1.0 - rate) + s * rate)


def normalize(x, stats):
    if stats is None:
        return x
    return (
        (x - tf.Variable(stats.mean, dtype=tf.float32)) /
        tf.math.sqrt(tf.Variable(stats.var, dtype=tf.float32))
    )


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean
