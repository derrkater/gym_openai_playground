import tensorflow as tf
import numpy as np
import gym
import tqdm

from tensorflow_util import variable_summaries

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')

    tf.reset_default_graph()

    # Forward
    inputs = tf.placeholder(shape=[1, 16], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
    variable_summaries(W)
    Q_out = tf.matmul(inputs, W)
    variable_summaries(Q_out)
    prediction = tf.argmax(Q_out, 1)

    # Loss
    Q_next = tf.placeholder(shape=[1, 4], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(Q_next - Q_out))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    update_model = trainer.minimize(loss)

    merged_summary = tf.summary.merge_all()
    init = tf.initialize_all_variables()

    gamma = .99
    epsilon = .1
    num_episodes = 2000

    steps = []
    rewards = []

    with tf.Session() as session:
        writer = tf.summary.FileWriter('tf_summaries', session.graph)
        session.run(init)
        # summary = session.run(merged_summary)
        # writer.add_summary(summary)

        for i_episode in tqdm.tqdm(range(num_episodes)):
            observation = env.reset()
            rewards_sum = 0
            t = reward = 0

            for t in range(100):

                onehot_state = np.identity(16)[observation: observation + 1]
                summary, greedy_action, all_Q = session.run([merged_summary, prediction, Q_out], feed_dict={inputs: onehot_state})
                writer.add_summary(summary, t)

                if np.random.rand(1) < epsilon:
                    greedy_action[0] = env.action_space.sample()

                observation_after, reward, done, _ = env.step(greedy_action[0])

                onehot_state_after = np.identity(16)[observation_after: observation_after + 1]
                Q = session.run(Q_out, feed_dict={inputs: onehot_state_after})
                Q_max = np.max(Q)
                Q_target = all_Q
                Q_target[0, greedy_action[0]] = reward + gamma * Q_max

                # onehot_state = np.identity(16)[observation: observation + 1]
                _, W_after = session.run([update_model, W], feed_dict={inputs: onehot_state, Q_next: Q_target})

                rewards_sum += reward
                observation = observation_after

                if done:
                    epsilon = 1. / (i_episode / 50. + 10.)
                    break

            steps.append(t)
            rewards.append(reward)

    sum(rewards) / num_episodes
