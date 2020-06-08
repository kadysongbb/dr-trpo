import argparse
import json
import os

import gym, roboschool
import numpy as np
import tensorflow as tf
from tqdm import trange

import utils.utils as utils
from noises.ounoise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from environment.wapper import Wrapper
from GAC.networks import AutoRegressiveStochasticActor as AIQN
from GAC.networks import StochasticActor as IQN
from GAC.networks import Critic, Value
from GAC.agent import GACAgent


def create_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--environment', default="LunarLanderContinuous-v2",
        help='name of the environment to run'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.99, metavar='G',
        help='discount factor for reward'
    )
    parser.add_argument(
        '--tau', type=float, default=5e-3, metavar='G',
        help='discount factor for model'
    )
    parser.add_argument('--noise', default='normal', choices=['ou', 'normal'])
    parser.add_argument('--noise_scale', type=float, default=0.2, metavar='G')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N')
    parser.add_argument('--epochs', type=int, default=None, metavar='N')
    parser.add_argument('--epoch_cycles', type=int, default=20, metavar='N')
    parser.add_argument('--rollout_steps', type=int, default=100, metavar='N')
    parser.add_argument(
        '--num_trainsteps', type=int, default=50, metavar='N',
        help='number of training steps'
    )
    parser.add_argument('--start_timesteps', type=int, default=1000, metavar='N')
    parser.add_argument('--eval_freq', type=int, default=500000, metavar='N')
    parser.add_argument('--eval_episodes', type=int, default=10, metavar='N')
    parser.add_argument(
        '--buffer_size', type=int, default=1000000, metavar='N',
        help='size of replay buffer'
    )
    parser.add_argument('--action_samples', type=int, default=16)
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument(
        '--experiment_name', default=None, type=str,
        help='provide an experiment name'
    )
    parser.add_argument('--print', default=False, action='store_true')
    parser.add_argument('--actor', default='AIQN', choices=['IQN', 'AIQN'])
    parser.add_argument(
        '--normalize_obs', default=False, action='store_true', 
        help='normalize observations'
    )
    parser.add_argument(
        '--normalize_rewards', default=False, action='store_true', 
        help='normalize rewards'
    )
    parser.add_argument(
        '--q_normalization', type=float, default=0.01,
        help='uniformly smooth the Q function in this range'
    )
    parser.add_argument(
        '--mode', type=str, default='boltzmann', choices=['linear', 'boltzmann', 'odrpo'],
        help='target policy is constructed based on this operator'
    )
    parser.add_argument(
        '--beta', type=float, default=1.0,
        help='dual parameter beta for normalizing actions'
    )
    parser.add_argument(
        '--num_steps', type=int, default=5000, metavar='N',
        help='number of training steps to play the environments game'
    )
    return parser


def _reset_noise(agent, a_noise):
    if a_noise is not None:
        a_noise.reset()


def evaluate_policy(policy, env, episodes):
    """
    Run the environment env using policy for episodes number of times.
    Return: average rewards per episode.
    """
    rewards = []
    for _ in range(episodes):
        state = np.float32(env.reset())
        is_terminal = False
        while not is_terminal:
            action = policy.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
            # remove the batch_size dimension if batch_size == 1
            action = tf.squeeze(action, [0]).numpy()
            state, reward, is_terminal, _ = env.step(action)
            state, reward = np.float32(state), np.float32(reward)
            rewards.append(float(reward))
            # env.render()
    return rewards


def main():
    print(tf.__version__)
    print("GPU Available: ", tf.test.is_gpu_available())

    args = create_argument_parser().parse_args()

    """
    Create Mujoco environment
    """
    env = Wrapper(gym.make(args.environment), args)
    eval_env = Wrapper(gym.make(args.environment), args)
    args.action_dim = env.action_space.shape[0]
    args.state_dim = env.observation_space.shape[0]

    if args.noise == 'ou':
        noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(env.action_space.shape[0]),
            sigma=float(args.noise_scale) * np.ones(env.action_space.shape[0])
        )
    elif args.noise == 'normal':
        noise = NormalActionNoise(
            mu=np.zeros(env.action_space.shape[0]),
            sigma=float(args.noise_scale) * np.ones(env.action_space.shape[0])
        )
    else:
        noise = None

    base_dir = os.getcwd() + '/models/' + args.environment + '/'
    run_number = 0
    while os.path.exists(base_dir + str(run_number)):
        run_number += 1
    base_dir = base_dir + str(run_number)
    os.makedirs(base_dir)
    gac = GACAgent(**args.__dict__)

    state = env.reset()
    results_dict = {
        'train_rewards': [],
        'eval_rewards': [],
        'actor_losses': [],
        'value_losses': [],
        'critic_losses': []
    }
    episode_steps, episode_rewards = 0, 0 # total steps and rewards for each episode

    num_steps = args.num_steps
    if num_steps is not None:
        nb_epochs = int(num_steps) // (args.epoch_cycles * args.rollout_steps)
    else:
        nb_epochs = 500

    _reset_noise(gac, noise)
    """
    training loop
    """
    average_rewards = 0
    count = 0
    total_steps = 0
    train_steps = 0
    for epoch in trange(nb_epochs):
        for cycle in range(args.epoch_cycles):
            for rollout in range(args.rollout_steps):
                """
                Get an action from neural network and run it in the environment
                """
                # print('t:', t)
                if total_steps < args.start_timesteps:
                    action = tf.expand_dims(env.action_space.sample(), 0)
                else:
                    action = gac.select_perturbed_action(
                        tf.convert_to_tensor([state], dtype=tf.float32),
                        noise
                    )
                # remove the batch_size dimension if batch_size == 1
                action = tf.squeeze(action, [0]).numpy()
                next_state, reward, is_terminal, _ = env.step(action)
                next_state, reward = np.float32(next_state), np.float32(reward)
                gac.store_transition(state, action, reward, next_state, is_terminal)
                episode_rewards += reward
                # print('average_rewards:', average_rewards)

                # check if game is terminated to decide how to update state, episode_steps,
                # episode_rewards
                if is_terminal:
                    state = np.float32(env.reset())
                    results_dict['train_rewards'].append(
                        (total_steps, episode_rewards)
                    )
                    with open('results.txt', 'w') as file:
                        file.write(json.dumps(results_dict))
                    episode_steps = 0
                    episode_rewards = 0
                    _reset_noise(gac, noise)
                else:
                    state = next_state
                    episode_steps += 1

                if total_steps % 100 == 0:
                    print('current progress: ' + str(total_steps))
                # evaluate
                if total_steps % args.eval_freq == 0:
                    eval_rewards = evaluate_policy(gac, eval_env, args.eval_episodes)
                    eval_reward = sum(eval_rewards) / args.eval_episodes
                    eval_variance = float(np.var(eval_rewards))
                    results_dict['eval_rewards'].append({
                        'total_steps': total_steps,
                        'train_steps': train_steps,
                        'average_eval_reward': eval_reward,
                        'eval_reward_variance': eval_variance
                    })
                    with open('results.txt', 'w') as file:
                        file.write(json.dumps(results_dict))
                total_steps += 1
            # train
            if gac.replay.size >= args.batch_size:
                for _ in range(args.num_trainsteps):
                    gac.train_one_step()
                    train_steps += 1

    with open('results.txt', 'w') as file:
        file.write(json.dumps(results_dict))

    utils.save_model(gac.actor, base_dir)


if __name__ == '__main__':
    main()
