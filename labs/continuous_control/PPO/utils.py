from collections import defaultdict
import numpy as np


class Trajectory(object):
    def __init__(self):
        self.data = defaultdict(list)
        self.score = 0.

    def add(self, states, rewards, probs, actions, values, dones):
        self.data['state'].append(states)
        self.data['reward'].append(rewards)
        self.data['prob'].append(probs)
        self.data['action'].append(actions)
        self.data['value'].append(values)
        self.data['done'].append(dones)
        self.score += np.mean(rewards)

    def __len__(self):
        return len(self.data['state'])

    def __getitem__(self, key):
        return self.data[key]


class Batcher:
    '''
    Select random batches from the dataset passed
    author: Shangtong Zhang
    source: https://bit.ly/2yrYoHy
    '''

    def __init__(self, batch_size, data):
        '''
        Initialize a Batcher object. sala all parameters as attributes
        :param batch_size: integer. The size of each batch
        :data: list. Dataset to be batched
        '''
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        '''
        start the counter
        '''
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        '''check if the dataset has been consumed'''
        return self.batch_start >= self.num_entries

    def next_batch(self):
        '''select the next batch'''
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size,
                             self.num_entries)
        return batch

    def shuffle(self):
        '''shuffle the datset passed in the constructor'''
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
