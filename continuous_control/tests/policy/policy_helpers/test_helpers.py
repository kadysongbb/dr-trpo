import tensorflow as tf

from policy.policy_helpers.helper_classes import ActionSampler
from policy.gac.networks import AutoRegressiveStochasticActor


def test_ActionSampler():
    asampler = ActionSampler(4)
    aractor = AutoRegressiveStochasticActor(3, 4, 5)
    print(asampler.get_actions(aractor, tf.convert_to_tensor([[1.,2.,3.]]), None))


if __name__ == '__main__':
    test_ActionSampler()
