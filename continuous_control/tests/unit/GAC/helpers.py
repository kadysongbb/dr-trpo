import unittest

import tensorflow as tf

import GAC.helpers as helpers


class TestHelpers(unittest.TestCase):

    def test_replay_buffer(self):
        state_dim = 4
        action_dim = 4
        size = 4
        replay_buffer = helpers.ReplayBuffer(state_dim, action_dim, size)
        state = [1, 2, 3, 4]
        action = [1, 2, 3, 4]
        reward = 1
        new_state = [1, 2, 3, 5]
        done = False
        replay_buffer.store(state, action, reward, new_state, done)
        self.assertEqual(replay_buffer.obs1_buf[0, 0], state[0])
        self.assertEqual(replay_buffer.obs1_buf[0, 1], state[1])
        self.assertEqual(replay_buffer.obs1_buf[0, 2], state[2])
        self.assertEqual(replay_buffer.obs1_buf[0, 3], state[3])

    def test_sampling_functionality(self):
        state_dim = 4
        action_dim = 4
        size = 4
        replay_buffer = helpers.ReplayBuffer(state_dim, action_dim, size)
        state = [1, 2, 3, 4]
        action = [1, 2, 3, 4]
        reward = 1
        new_state = [1, 2, 3, 5]
        done = False
        batch = 2
        replay_buffer.store(state, action, reward, new_state, done)
        replay_buffer.store(state, action, reward, new_state, done)
        replay_buffer.store(state, action, reward, new_state, done)
        replay_buffer.store(state, action, reward, new_state, done)
        replay_buffer.store(state, action, reward, new_state, done)
        replay_buffer.store(state, action, reward, new_state, done)
        sample = replay_buffer.sample_batch(batch)
        self.assertEqual(sample.s[0, 0], state[0])
        self.assertEqual(sample.s[0, 1], state[1])
        self.assertEqual(sample.s[0, 2], state[2])
        self.assertEqual(sample.s[0, 3], state[3])


if __name__ == '__main__':
    unittest.main()
