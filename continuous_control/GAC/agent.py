# import external dependencies
import numpy as np
import tensorflow as tf

# import local dependencies
from GAC.networks import StochasticActor, AutoRegressiveStochasticActor,  Critic, Value
from GAC.helpers import ReplayBuffer, update, ActionSampler, normalize, denormalize
from utils.utils import RunningMeanStd


"""
File Description:

Class: Generative Actor Critic (GAC) agent.
"""


class GACAgent:
    """
    GAC agent.
    Action is always from -1 to 1 in each dimension.
    """
    def __init__(self, action_dim, state_dim, buffer_size=1000000, action_samples=10,
                 mode='linear', beta=1, tau=5e-3, q_normalization=0.01, gamma=0.99,
                 normalize_obs=False, normalize_rewards=False, batch_size=64, actor='AIQN',
                 *args, **kwargs):
        """
        Agent class to generate a stochastic policy.

        Args:
            action_dim (int): action dimension
            state_dim (int): state dimension
            buffer_size (int): how much memory is allocated to the ReplayMemoryClass
            action_samples (int): originally labelled K in the paper, represents how many
                actions should be sampled from the memory buffer
            mode (string): poorly named variable to represent variable being used in the
                distribution being used
            beta (float): value used in boltzmann distribution
            tau (float): update rate parameter
            batch_size (int): batch size
            q_normalization (float): q value normalization rate
            gamma (float): value used in critic training
            normalize_obs (boolean): boolean to indicate that you want to normalize
                observations
            normalize_rewards (boolean): boolean to indicate that you want to normalize
                return values (usually done for numerical stability)
            actor (string): string indicating the type of actor to use
        """
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.action_samples = action_samples
        self.mode = mode
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size

        # normalization
        self.normalize_observations = normalize_obs
        self.q_normalization = q_normalization
        self.normalize_rewards = normalize_rewards

        # type of actor being used
        if actor == 'IQN':
            self.actor = StochasticActor(self.state_dim, self.action_dim)
            self.target_actor = StochasticActor(self.state_dim, self.action_dim)
        elif actor == 'AIQN':
            self.actor = AutoRegressiveStochasticActor(self.state_dim, self.action_dim)
            self.target_actor = AutoRegressiveStochasticActor(self.state_dim, self.action_dim)

        if self.normalize_observations:
            self.obs_rms = RunningMeanStd(shape=self.state_dim)
        else:
            self.obs_rms = None

        if self.normalize_rewards:
            self.ret_rms = RunningMeanStd(shape=1)
            self.ret = 0
        else:
            self.ret_rms = None

        # initialize trainable variables
        self.actor(
            tf.zeros([self.batch_size, self.state_dim]),
            tf.zeros([self.batch_size, self.action_dim])
        )
        self.target_actor(
            tf.zeros([self.batch_size, self.state_dim]),
            tf.zeros([self.batch_size, self.action_dim])
        )

        self.critics = Critic(self.state_dim, self.action_dim)
        self.target_critics = Critic(self.state_dim, self.action_dim)

        # initialize trainable variables for critics
        self.critics(
            tf.zeros([self.batch_size, self.state_dim]),
            tf.zeros([self.batch_size, self.action_dim])
        )
        self.target_critics(
            tf.zeros([self.batch_size, self.state_dim]),
            tf.zeros([self.batch_size, self.action_dim])
        )

        self.value = Value(self.state_dim)
        self.target_value = Value(self.state_dim)

        # initialize value training variables
        self.value(tf.zeros([self.batch_size, self.state_dim]))
        self.value(tf.zeros([self.batch_size, self.state_dim]))

        # initialize the target networks.
        update(self.target_actor, self.actor, 1.0)
        update(self.target_critics, self.critics, 1.0)
        update(self.target_value, self.value, 1.0)

        self.replay = ReplayBuffer(self.state_dim, self.action_dim, self.buffer_size)
        self.action_sampler = ActionSampler(self.actor.action_dim)

    def train_one_step(self):
        """
        Execute one update for each of the networks. Note that if no positive advantage elements
        are returned the algorithm doesn't update the actor parameters.

        Args:
            None

        Returns:
            None
        """
        # transitions is sampled from replay buffer
        transitions = self.replay.sample_batch(self.batch_size)
        state_batch = normalize(transitions.s, self.obs_rms)
        action_batch = transitions.a
        reward_batch = normalize(transitions.r, self.ret_rms)
        next_state_batch = normalize(transitions.sp, self.obs_rms)
        terminal_mask = transitions.it
        # transitions is sampled from replay buffer
        self.critics.train(
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            terminal_mask,
            self.target_value,
            self.gamma,
            self.q_normalization
        )
        self.value.train(
            state_batch,
            self.target_actor,
            self.target_critics,
            self.action_samples
        )
        # note that transitions.s represents the sampled states from the memory buffer
        states, actions, advantages = self._sample_positive_advantage_actions(state_batch)
        if advantages.shape[0]:
            self.actor.train(
                states,
                actions,
                advantages,
                self.mode,
                self.beta
            )
        update(self.target_actor, self.actor, self.tau)
        update(self.target_critics, self.critics, self.tau)
        update(self.target_value, self.value, self.tau)

    def _sample_positive_advantage_actions(self, states):
        """
        Sample from the target network and a uniform distribution.
        Then only keep the actions with positive advantage.
        Returning one action per state, if more needed, make states contain the
        same state multiple times.

        Args:
            states (tf.Variable): states of dimension (batch_size, state_dim)

        Returns:
            good_states (list): Set of positive advantage states (batch_size, sate_dim)
            good_actions (list): Set of positive advantage actions
            advantages (list[float]): set of positive advantage values (Q - V)
        """
        # tile states to be of dimension (batch_size * K, state_dim)
        tiled_states = tf.tile(states, [self.action_samples, 1])
        # Sample actions with noise for regularization
        target_actions = self.action_sampler.get_actions(self.target_actor, tiled_states)
        target_actions += tf.random.normal(target_actions.shape) * 0.01
        target_actions = tf.clip_by_value(target_actions, -1, 1)
        target_q = self.target_critics(tiled_states, target_actions)
        # Sample multiple actions both from the target policy and from a uniform distribution
        # over the action space. These will be used to determine the target distribution
        random_actions = tf.random.uniform(target_actions.shape, minval=-1.0, maxval=1.0)
        random_q = self.target_critics(tiled_states, random_actions)
        # create target actions vector, consistent of purely random actions and noisy actions
        # for the sake of exploration
        target_actions = tf.concat([target_actions, random_actions], 0)
        # compute Q and V values with dimensions (2 * batch_size * K, 1)
        q = tf.concat([target_q, random_q], 0)
        # determine the estimated value of a given state
        v = self.target_value(tiled_states)
        v = tf.concat([v, v], 0)
        # expand tiled states to allow for indexing later on
        tiled_states = tf.concat([tiled_states, tiled_states], 0)
        # remove unused dimensions
        q_squeezed = tf.squeeze(q)
        v_squeezed = tf.squeeze(v)
        # select s, a with positive advantage
        squeezed_indicies = tf.where(q_squeezed > v_squeezed)
        # collect all advantegeous states and actions
        good_states = tf.gather_nd(tiled_states, squeezed_indicies)
        good_actions = tf.gather_nd(target_actions, squeezed_indicies)
        # retrieve advantage values
        advantages = tf.gather_nd(q-v, squeezed_indicies)
        return good_states, good_actions, advantages

    def get_action(self, states):
        """
        Get a set of actions for a batch of states

        Args:
            states (tf.Variable): dimensions (batch_size, state_dim)

        Returns:
            sampled actions for the given state with dimension (batch_size, action_dim)
        """
        return self.action_sampler.get_actions(self.actor, states)

    def select_perturbed_action(self, state, action_noise=None):
        """
        Select actions from the perturbed actor using action noise and parameter noise

        Args:
            state (tf.Variable): tf variable containing the state vector
            action_niose (function): action noise function which will construct noise from some
                distribution

        Returns:
            action vector of dimension (batch_size, action_dim). Note that if action noise,
                this function is the same as get_action.
        """
        state = normalize(tf.Variable(state, dtype=tf.float32), self.obs_rms)
        action = self.action_sampler.get_actions(self.actor, state)
        if action_noise is not None:
            action += tf.Variable(action_noise(), dtype=tf.float32)
        action = tf.clip_by_value(action, -1, 1)
        return action

    def store_transition(self, state, action, reward, next_state, is_done):
        """
        Store the transition in the replay buffer with normalizing, should it be specified.

        Args:
            state (tf.Variable): (batch_size, state_size) state vector
            action (tf.Variable): (batch_size, action_size) action vector
            reward (float): reward value determined by the environment (batch_size, 1)
            next_state (tf.Variable): (batch_size, state_size) next state vector
            is_done (boolean): value to indicate that the state is terminal
        """
        self.replay.store(state, action, reward, next_state, is_done)
        if self.normalize_observations:
            self.obs_rms.update(state)
        if self.normalize_rewards:
            self.ret = self.ret * self.gamma + reward
            self.ret_rms.update(np.array([self.ret]))
            if is_done:
                self.ret = 0
