import tensorflow as tf

from policy.gac.networks import Value, Critic, AutoRegressiveStochasticActor as Actor
from policy.policy_helpers.helper_classes import ActionSampler, Transition


actor = Actor(3, 4, 5)
critic = Critic(3+4, 2)
value = Value(3, 1)
sampler = ActionSampler(4)

transitions = Transition(
        tf.convert_to_tensor(
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
        ),
        tf.convert_to_tensor(
        [
            [0., 0., 0., 0.],
            [0.1, 0.2, 0.3, 0.1],
            [0.4, 0.5, 0.6, 0.9],
        ]
        ),
        tf.convert_to_tensor(
        [
            1., 2., 3.
        ]
        ),
        tf.convert_to_tensor(
        [
            [4., 5., 6.],
            [7., 8., 9.],
            [1., 2., 3.],
        ]
        ),
        tf.convert_to_tensor(
        [
            False, False, False
        ]
        ),
    )

train_history = value.train(transitions, sampler, actor, critic, 2)
print(train_history.history)
