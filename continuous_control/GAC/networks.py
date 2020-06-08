import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.ops import gen_array_ops

"""
File Description:

This file is meant to contain all networks pertinent to the GAC algorithm. This includes the
Stochastic actors (AIQN and IQN), the value network, and the critic network. Note that, while they
maintain the same architecture, it is advisable that the value and critic netwok be DIFFERENT
classes, if for no other reason than legibility.
"""


class CosineBasisLinear(tf.Module):
    def __init__(self, n_basis_functions, embed_dim, activation = None):
        """
        Parametrize the embedding function using Fourier series up to n_basis_functions terms.
        It's an entry-wise embedding function, i.e. from R to R^d. Could do nonlinear transform
        with activation in the end.

        Class Args:
            n_basis_functions (int): the number of basis functions
            embed_dim (int): the dimensionality of embedding
            activation (tf.function): activation function.
        """
        super(CosineBasisLinear, self).__init__()
        # coefficient of the basis
        self.act_linear = Dense(embed_dim, activation = activation, input_shape = (n_basis_functions,))
        self.n_basis_functions = n_basis_functions
        self.embed_dim = embed_dim

    def _cosine_basis_functions(self, x, n_basis_functions=64):
        """
        Cosine basis function (the function is denoted as psi in the paper). This is used to embed
        [0, 1] -> R^d. The i th component of output is cos(i*x).

        Args:
            x (tf.Variable)
            n_basis_functions (int): number of basis function for the
        """
        x = tf.reshape(x, (-1, 1))
        i_pi = np.tile(np.arange(1, n_basis_functions + 1, dtype=np.float32), (x.shape[0], 1)) * np.pi
        i_pi = tf.convert_to_tensor(i_pi)
        embedding = tf.math.cos(x * i_pi)
        return embedding

    def __call__(self, x):
        """
        Args:
            x: tensor (batch_size, a), a is arbitrary, e.g. dimensionality of action vector.
        Return:
            out: tensor (batch_size, a, embed_dim): the embedding vector phi(x).
        """
        batch_size = x.shape[0]
        h = self._cosine_basis_functions(x, self.n_basis_functions)
            # (size of x , n_basis_functions)
        out = self.act_linear(h) # (size of x , embed_dim)
        out = tf.reshape(out, (batch_size, -1, self.embed_dim)) #(batch_size, a, embed_dim)
        return out


class IQNActor(tf.Module):
    def __init__(self, state_dim, action_dim):
        super(IQNActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # we will use in the loss function.

        self.module_type = 'IQNActor'
        self.huber_loss_function = tf.keras.losses.Huber(
            delta=1.0,
            reduction=tf.keras.losses.Reduction.NONE
        )  # delta is kappa in paper
        self.optimizer = tf.keras.optimizers.Adam(0.0001)

    def target_density(self, mode, advantage, beta, prob):
        """
        The density of target policy D(a|s). Comes from table 1 in the paper.

        Args:
            mode: ["linear", "boltzmann", "odrpo"]
            advantange: (batch_size, 1)
        Returns:
            density of D(a|s)

        """
        if mode == "linear":
            return advantage / tf.reduce_sum(advantage)
        elif mode == "boltzmann":
            return tf.nn.softmax(advantage/beta)
        elif mode == "odrpo":
            numerator =  tf.math.reduce_max(tf.math.multiply(tf.exp(advantage/beta), tf.cast(prob, tf.float32)), axis=1,keepdims=True)
            denominator = tf.reduce_sum(numerator)
            return numerator/denominator
        else:
            raise NotImplementedError

    def huber_quantile_loss(self, actions, target_actions, taus, weights):
        """
        Compute Huber losses for quantile regression.

        rho function in the paper = |taus - (target_actions - action) < 0| * huber_loss

        Args:
            actions (tf.Tensor): (batch_size, action_dim), Quantile prediction from taus
            target_actions (tf.Tensor): (batch_size, action_dim)
            taus (tf.Variable): (batch_size, action_dim)
            weights (tf.Variable): (batch_size, 1), The density of target action distribution D(a|s)

        Returns:
            Huber quantile loss
        """

        I_delta = tf.dtypes.cast(((actions - target_actions) > 0), tf.float32)
        eltwise_huber_loss = self.huber_loss_function(target_actions, actions)
        # delta is kappa in paper
        eltwise_loss = tf.math.abs(taus - I_delta) * eltwise_huber_loss * weights
        #(batch_size, action_dim)

        # mean over batches, sum over action dimensions, according to Algorithm 2.
        return tf.math.reduce_mean(eltwise_loss)

    def train(self, states, supervise_actions, advantage, mode, beta):
        """
        the batch_size here combines the state_batch_size and action samples.

        Args:
            states: (batch_size, state_dim)
            supervise_actions: (batch_size, action_dim)
            advantage: (batch_size, 1)
            mode (string): the type of distribution being used
            beta (float): update rate for the Actor
        """

        taus = tf.random.uniform(tf.shape(supervise_actions))
        # ODRPO mode requires previous policy probability
        unique_actions, idx, count = gen_array_ops.unique_with_counts_v2(supervise_actions, [0])
        num_action_samples = len(idx)
        prob = tf.Variable(tf.zeros(num_action_samples, 1))
        for i in range(num_action_samples):
            prob = prob[i].assign(tf.cast(count[idx[i]]/num_action_samples, tf.float32))

        weights = self.target_density(mode, advantage, beta, prob)

        with tf.GradientTape() as tape:
            actions = self(states, taus, supervise_actions) #(batch_size, action_dim)
            loss = self.huber_quantile_loss(actions, supervise_actions, taus, weights)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AutoRegressiveStochasticActor(IQNActor):
    def __init__(self, state_dim, action_dim, n_basis_functions = 64):
        """
        the autoregressive stochastic actor is an implicit quantile network used to sample from a
        distribution over optimal actions. The model maintains it's autoregressive quality due to
        the recurrent network used.

        Class Args:
            state_dim (int): the dimensionality of the state vector
            action_dim (int): the dimensionality of the action vector
            n_basis_functions (int): the number of basis functions
        """
        super(AutoRegressiveStochasticActor, self).__init__(state_dim, action_dim)
        # create all necessary class variables
        self.module_type = 'AutoRegressiveStochasticActor'
        self.state_embedding = Dense(
            400,  # as specified by the architecture in the paper and in their code
            activation=tf.keras.layers.LeakyReLU(alpha=0.01)
        )
        # use the cosine basis linear classes to "embed" the inputted values to a set dimension
        # this is equivalent to the psi function specified in the Actor diagram
        self.noise_embedding = CosineBasisLinear(n_basis_functions, 400)
        self.action_embedding = CosineBasisLinear(n_basis_functions, 400)

        # construct the GRU to ensure autoregressive qualities of our samples
        self.rnn = tf.keras.layers.GRU(400, return_state=True, return_sequences=True)
        # post processing linear layers
        self.dense_layer_1 = Dense(400, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
        # output layer (produces the sample from the implicit quantile function)
        # note the output is between [0, 1]
        self.dense_layer_2 = Dense(1, activation=tf.nn.tanh)

    def __call__(self, state, taus, supervise_actions=None):
        """
        Analogous to the traditional call function in most models. This function conducts a single
        forward pass of the AIQN given the state.

        Args:
            state (tf.Variable): state vector containing a state with the format R^state_dim
            taus (tf.Variable): randomly sampled noise vector for sampling purposes. This vector
                should be of shape (batch_size x actor_dimension)
            supervise_actions (tf.Variable): set of previous actions

        Returns:
            actions vector
        """
        if supervise_actions is not None:
            # if the actions are defined then we use the supervised forward method which generates
            # actions based on the provided sequence
            return self._supervised_forward(state, taus, supervise_actions)
        batch_size = state.shape[0]
        # batch x 1 x 400
        state_embedding = tf.expand_dims(tf.nn.leaky_relu(self.state_embedding(state)), 1)
        # batch x action dim x 400
        noise_embedding = self.noise_embedding(taus)

        action_list = []

        # allocate memory for the actions
        action = tf.zeros(batch_size, 1)
        hidden_state = None

        # If the prior actions are not provided then we generate the action vector dimension by
        # dimension. Note that the actions are in the domain [0, 1] (Why? I dunno).
        for idx in range(self.action_dim):
            # batch x 1 x 400
            action_embedding = tf.nn.leaky_relu(
                self.action_embedding(tf.reshape(action, (batch_size, 1, 1)))
            )
            rnn_input = tf.concat([state_embedding, action_embedding], axis=2)
            # Note that the RNN states encode the function approximation for the conditional
            # probability of the ordered sequence of vectors in d dimension space. Effectively,
            # the researchers claim that each variable in the d dimension vector are autocorrelated.
            gru_out, hidden_state = self.rnn(rnn_input, hidden_state)

            # batch x 400
            hadamard_product = tf.squeeze(gru_out, 1) * noise_embedding[:, idx, :]
            action = self.dense_layer_2(self.dense_layer_1(hadamard_product))
            action_list.append(action)

        actions = tf.squeeze(tf.stack(action_list, axis=1), -1)
        return actions

    def _supervised_forward(self, state, taus, supervise_actions):
        """
        Private function to conduct a supervised forward call. This is relying on the assumption
        actions are not independent to each other. With this assumption of "autocorrelation" between
        action dimensions, this function creates a new action vector using prior actions as input.

        Args:
            state (tf.Variable(array)): state vector representation
            taus (tf.Variable(array)): noise vector
            supervise_actions (tf.Variable(array)): actions vector (batch x action dim)

        Returns:
            a action vector of size (batch x action dim)
        """
        # F.leaky_relu(self.state_embedding(state)).unsqueeze(1).expand(-1, self.action_dim, -1)
        # batch x action dim x 400
        state_embedding = tf.expand_dims(tf.nn.leaky_relu(self.state_embedding(state)), 1)
        state_embedding = tf.broadcast_to(
            state_embedding,
            (
                state_embedding.shape[0],
                self.action_dim,
                state_embedding.shape[2]
            )
        )
        # batch x action dim x 400
        shifted_actions = tf.Variable(tf.zeros_like(supervise_actions))
        # assign shifted actions
        shifted_actions = shifted_actions[:, 1:].assign(supervise_actions[:, :-1])
        provided_action_embedding = tf.nn.leaky_relu(self.action_embedding(shifted_actions))

        rnn_input = tf.concat([state_embedding, provided_action_embedding], axis=2)
        gru_out, _ = self.rnn(rnn_input)

        # batch x action dim x 400
        noise_embedding = self.noise_embedding(taus)
        # batch x action dim x 400
        # take the element wise product of these vectors
        hadamard_product = gru_out * noise_embedding
        actions = self.dense_layer_2(self.dense_layer_1(hadamard_product))
        # batch x action dim
        return tf.squeeze(actions, -1)


class StochasticActor(IQNActor):
    def __init__(self, state_dim, action_dim, n_basis_functions=64):
        """
        The IQN stochasitc action generator, takes state and tau (random vector) as input, and output
        the next action. This generator is not in an autoregressive way, i.e. the next action is
        generated as a whole, instead of one dimension by one dimension.

        Class Args:
            state_dim (int): the dimensionality of the state vector
            action_dim (int): the dimensionality of the action vector
            n_basis_functions (int): the number of basis functions for noise embedding.
        """
        super(StochasticActor, self).__init__(state_dim, action_dim)
        self.module_type = 'StochasticActor'
        self.noise_embed_dim = 400 // action_dim

        self.state_embedding_layer = Dense(
            self.noise_embed_dim * self.action_dim,
            activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            input_shape = (self.state_dim,))

        self.noise_embedding_layer = CosineBasisLinear(
            n_basis_functions, self.noise_embed_dim,
            activation=tf.keras.layers.LeakyReLU(alpha=0.01))

        self.merge_embedding_layer = Dense(
            200, activation= tf.keras.layers.LeakyReLU(alpha=0.01),
            input_shape = (self.noise_embed_dim * self.action_dim,))

        self.output_action_layer = Dense(
            self.action_dim, activation= tf.nn.tanh,
            input_shape = (200,))

    def __call__(self, states, taus, supervise_actions = None):
        """
        Args:
            states: tensor (batch_size, state_dim)
            taus: tensor (batch_size, action_dim)
            supervise_actions = None: to be consistent with AIQN. But is not used here.
        Return:
            next_actions: tensor (batch_size, action_dim)
        """
        state_embedding = self.state_embedding_layer(states)
        # (batch_size, self.noise_embed_dim * self.action_dim)
        noise_embedding = self.noise_embedding_layer(taus)
        # (batch_size, self.action_dim, self.embed_dim)
        noise_embedding = tf.reshape(noise_embedding, (-1, self.noise_embed_dim * self.action_dim))
        # (batch_size, self.noise_embed_dim * self.action_dim)
        merge = state_embedding * noise_embedding
        # (batch_size, self.noise_embed_dim * self.action_dim)
        merge_embedding = self.merge_embedding_layer(merge)  #(batch_size, 200)
        actions = self.output_action_layer(merge_embedding) # (batch_size, self.action_dim)
        return actions


class FNN(tf.Module):

    def __init__(self, arch):
        super(FNN, self).__init__()
        self.layers = self._build_fnn_model(arch, activation=None)

    def _build_fnn_model(self, arch, activation=None):
        """
        Args:
            arch: A list of integers discribing the width of each layer.

        Returns:
            A list of layers.
        """
        if activation is None:
            activation = tf.keras.layers.LeakyReLU(alpha=0.01)

        layers = []
        for i in range(len(arch)-2):
            layers.append(Dense(arch[i+1], activation=activation, input_shape = (arch[i],)))
        layers.append(Dense(arch[-1], input_shape = (arch[-2],))) # the last layer don't need activation.

        return layers

    def __call__(self, x):
        '''
        layers: a list of layers
        '''
        length = len(self.layers)
        out = x
        for i in range(length):
            out = self.layers[i](out)
        return out


class Critic(tf.Module):
    '''
    The Critic class create one or two critic networks, which take states as input and return
    the value of those states. The critic has two hidden layers and an output layer with size
    400, 300, and 1. All are fully connected layers.

    Note that this is a black box critic which contains two networks.
    And we will always output the smaller predictions.
    Double critic trick.

    Class Args:
    state_dim (int): dim of states
    action_dim (int): dim of actions
    '''
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fnn1 = FNN([state_dim + action_dim, 400, 300, 1])
        self.fnn2 = FNN([state_dim + action_dim, 400, 300, 1])
        self.optimizer1 = tf.keras.optimizers.Adam(0.0001)
        self.optimizer2 = tf.keras.optimizers.Adam(0.0001)

    def __call__(self, states, actions):
        x = tf.concat([states, actions], -1)
        pred1 = self.fnn1(x)
        pred2 = self.fnn2(x)
        return tf.minimum(pred1, pred2)

    def train(self, states, actions, rewards, next_states, terminal_mask, value, gamma,
              q_normalization=0.01):
        """
        transitions is of type named tuple policy.policy_helpers.helpers.Transition
        q1, q2 are seperate Q networks, thus can be trained separately

        Args:
            states
            actions
            rewards
            next_states
            terminal_mask
            transitions
            value
            gamma

        Returns:
            critic history tuple (two histories for the two critic models in general)
        """
        # Add tau random noise sampler for Q value regularization
        batch_size = states.shape[0]
        noise = (tf.random.uniform((batch_size, self.action_dim), 0, 1) * 2 - 1) * q_normalization
        noisy_actions = actions + noise
        action_batch = tf.clip_by_value(noisy_actions, -1, 1)
        # Line 10 of Algorithm 2
        yQ = rewards + gamma * value(next_states) * (1 - terminal_mask)
        # Line 11-12 of Algorithm 2
        x = tf.concat([states, action_batch], -1)
        with tf.GradientTape() as tape1:
            loss1 = tf.keras.losses.mse(yQ, self.fnn1(x))
        gradients1 = tape1.gradient(loss1, self.fnn1.trainable_variables)
        self.optimizer1.apply_gradients(zip(gradients1, self.fnn1.trainable_variables))

        with tf.GradientTape() as tape2:
            loss2 = tf.keras.losses.mse(yQ, self.fnn2(x))
        gradients2 = tape2.gradient(loss2, self.fnn2.trainable_variables)
        self.optimizer2.apply_gradients(zip(gradients2, self.fnn2.trainable_variables))


class Value(tf.Module):

    """
    Value network has the same architecture as Critic
    """

    def __init__(self, state_dim):
        super(Value, self).__init__()
        self.state_dim = state_dim
        self.fnn = FNN([state_dim, 400, 300, 1])
        self.optimizer = tf.keras.optimizers.Adam(0.0001)

    def __call__(self, states):
        return self.fnn(states)

    def train(self, states, actor, critic, action_samples = 8):
        """
        transitions is of type named tuple policy.policy_helpers.helpers.Transition
        action_sampler is of type policy.policy_helpers.helpers.ActionSampler

        Args:
            states
            transitions
            actor
            critic
            action_samples

        Returns:
            Value history element.
        """
        # Each state needs action_samples action samples
        # originally, transitions.s is [batch_size , state_dim]
        # now [batch_size x action_samples , state dim]
        # we tiled in this way, so that after reshape we get back in the same order. Allowing for
        # the average reduce_mean function to work later
        tiled_states = tf.expand_dims(states, 1) # [batch_size , 1 , state_dim]
        tiled_states = tf.tile(tiled_states, [1, action_samples, 1])
        # [batch_size, action_samples, state dim]
        tiled_states = tf.reshape(tiled_states, [-1, self.state_dim])

        """
        Line 13 of Algorithm 2.
        Sample actions from the actor network given current state and tau ~ U[0,1].
        """
        taus = tf.random.uniform(shape=(tiled_states.shape[0], actor.action_dim))
        actions = actor(tiled_states, taus)

        """
        Line 14 of Algorithm 2.
        Get the Q value of the states and action samples.
        Average over all action samples for Q1, Q2 and take the minimum.
        Typo in Algorithm 2 line 14. 1/K is missed
        """

        Q = critic(tiled_states, actions)
        Q = tf.reshape(Q, [-1, action_samples, 1]) #(batch_size, action_samples, 1)
        v_critic = tf.reduce_mean(Q, 1) #(batch_size, 1)

        """
        Line 15 of Algorithm 2.
        Get value of current state from the Value network.
        Loss is MSE.
        """
        estimated_values = self
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.mse(v_critic, self(states))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
