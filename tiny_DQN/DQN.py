from __future__ import division, print_function, unicode_literals

# Handle arguments (before slow imports so --help can be fast)
import argparse
parser = argparse.ArgumentParser(
    description="Train a DQN net to play MsMacman.")
parser.add_argument("-n", "--number-steps", type=int, default=4000000,
    help="total number of training steps")
parser.add_argument("-s", "--save-steps", type=int, default=1000,
    help="number of training steps between saving checkpoints")
parser.add_argument("-l", "--learn-iterations", type=int, default=20,
    help="number of game iterations between each training step")
parser.add_argument("-p", "--path", default="my_dqn.ckpt",
    help="path of the checkpoint file")
parser.add_argument("-t", "--test", action="store_true", default=False,
    help="test (no learning and minimal epsilon)")
parser.add_argument("-v", "--verbosity", action="count", default=0,
    help="increase output verbosity")
args = parser.parse_args()

from collections import deque
import numpy as np
import os
import tensorflow as tf
from simulation import Simulation

#env = gym.make("MsPacman-v0")
env=Simulation()
done = True  # env needs to be reset
n_outputs=2

# First let's build the DQN
def q_network(X_state):
    inp = tf.layers.dense(X_state,4,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name="input")
    hidden = tf.layers.dense(inp,4,kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    outputs = tf.layers.dense(hidden,n_outputs,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name="output")
    #hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,activation=hidden_activation,kernel_initializer=initializer)
    #outputs = tf.layers.dense(hidden, n_outputs,kernel_initializer=initializer)
    return outputs

X_state = tf.placeholder(tf.float32, shape=[None,5])
online_q_values = q_network(X_state)

# Now for the training operations
learning_rate = 0.001
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keep_dims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Let's implement a simple replay memory
replay_memory_size = 20000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
           cols[4].reshape(-1, 1))

# And on to the epsilon-greedy policy with decaying epsilon
eps_min = 0.1
eps_max = 1.0 if not args.test else eps_min
eps_decay_steps = args.number_steps // 2

def epsilon_greedy(q_values,state,sim,step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    # Account for deterministic actions first
    if sim.is_next_action_deterministic(state):
        return sim.get_next_deterministic_action(state)
    elif np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

# TensorFlow - Execution phase
training_start = 10000  # start training after 10,000 game iterations
discount_rate = 0.6
skip_start = 90  # Skip the start of every game (it's just waiting time).
batch_size = 50
iteration = 0  # game iterations
done = 1 # env needs to be reset

# We will keep track of the max Q-Value over time and compute the mean per game
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0
loss_list=[]
with tf.Session() as sess:
    if os.path.isfile(args.path + ".index"):
        saver.restore(sess, args.path)
    else:
        init.run()
    while True:
        step = global_step.eval()
        if step >= args.number_steps:
            break
        iteration += 1
        if args.verbosity > 0:
            print("\rIteration {}   Training step {}/{} ({:.1f})%   "
                  "Loss {:5f}    Mean Max-Q {:5f}   ".format(
            iteration, step, args.number_steps, step * 100 / args.number_steps,
            loss_val, mean_max_q), end="")
            loss_list.append(loss_val)
        if done==1: # game over, start again
            env.prev_u_t=9.0
            state,done = env.get_first_state()
        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values,state,env,step)
        # Online DQN plays
        next_state,done=env.get_next_state(state,action)
        reward=env.get_reward(next_state,done,action)
        # Let's memorize what happened
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        # Compute statistics for tracking progress
        total_max_q += q_values.max()
        game_length += 1
        if done==1:
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length = 0

        if iteration < training_start or iteration % args.learn_iterations != 0:
            continue # only train after warmup period and at regular intervals

        # Sample memories and use the target DQN to produce the target Q-Value
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            sample_memories(batch_size))
        next_q_values = online_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Train the online DQN
        _, loss_val = sess.run([training_op, loss], feed_dict={
            X_state: X_state_val, X_action: X_action_val, y: y_val})

        # And save regularly
        if step % args.save_steps == 0:
            saver.save(sess, args.path)
