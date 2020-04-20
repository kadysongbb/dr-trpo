import tensorflow as tf

from policy.gac.networks import Value, Critic
from policy.policy_helpers.helper_classes import Transition


critic = Critic(3+4, 2)
value = Value(3, 1)

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

history1, history2 = critic.train(transitions, value, 0.99)
print(history1.history, history2.history)
