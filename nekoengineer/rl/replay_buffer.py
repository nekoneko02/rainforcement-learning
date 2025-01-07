import numpy as np
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import tensorflow as tf
from tensorflow import keras
import math

class Buffer(tf_uniform_replay_buffer.TFUniformReplayBuffer):
  def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64):
    data_spec =  (
      tf.TensorSpec([num_states], tf.float32, 'observation'),
      tf.TensorSpec([num_actions], tf.float32, 'action'),
      tf.TensorSpec([1], tf.float32, 'reward'),
      tf.TensorSpec([num_states], tf.float32, 'next_observation'),
      tf.TensorSpec([1], tf.float32, 'priority_key')
    )
    self.priority_key = 4

    super().__init__(
        data_spec=data_spec,
        batch_size=batch_size,
        max_length=buffer_capacity
    )
  def get_next(self, sample_batch_size=None, num_steps=None, time_stacked=True):
    samples, info = super().get_next(sample_batch_size=self.max_length, num_steps=num_steps, time_stacked=time_stacked)

    priority_batch = samples[self.priority_key]
    probs = tf.nn.softmax(priority_batch)
    indices = tf.random.categorical(math.log(probs), num_samples=sample_batch_size)
    samples = samples[indices]

    return samples, info
    """
    # Number of "experiences" to store at max
    self.buffer_capacity = buffer_capacity
    # Num of tuples to train on.
    self.batch_size = batch_size

    # Its tells us num of times record() was called.
    self.buffer_counter = 0

    # Instead of list of tuples as the exp.replay concept go
    # We use different np.arrays for each tuple element
    self.state_buffer = np.zeros((self.buffer_capacity, num_states))
    self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
    self.reward_buffer = np.zeros((self.buffer_capacity, 1))
    self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

  # Takes (s,a,r,s') observation tuple as input
  def record(self, obs_tuple):
    # Set index to zero if buffer_capacity is exceeded,
    # replacing old records
    index = self.buffer_counter % self.buffer_capacity

    self.state_buffer[index] = obs_tuple[0]
    self.action_buffer[index] = obs_tuple[1]
    self.reward_buffer[index] = obs_tuple[2]
    self.next_state_buffer[index] = obs_tuple[3]

    self.buffer_counter += 1
    """