import logging
import time
import gym
import numpy as np

import tensorflow as tf

if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    # env = gym.make('FrozenLake-v0')

    print(env.action_space)
    print(env.observation_space)
    print(env.reward_range)

    # Q = np.zeros([env.observation_space.n, env.action_space.n])

    for i_episode in range(20):
        print('EPISODE {}'.format(i_episode))
        observation = env.reset()
        for t in range(100):
            env.render()
            observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
            if done:
                break
