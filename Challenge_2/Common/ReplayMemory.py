import numpy as np
import matplotlib.pyplot as plt


class ReplayMemory():
    def __init__(self, capacity, entry_size):
        """
        Create the replay memory.

        :param capacity: the total capacity (old values will be overwritten)
        :param entry_size: the size of a single entry
        """
        self.memory = np.empty((capacity, entry_size))
        self.next_index = 0
        self.capacity = capacity
        self.entry_size = entry_size

        self.valid_entries = 0

    def push(self, entry):
        """
        Push an entry to the memory.

        :param entry: the entry to push
        """
        self.memory[self.next_index] = entry
        self.next_index = (self.next_index + 1) % self.capacity

        if self.valid_entries < self.capacity:
            self.valid_entries += 1

    def sample(self, count):
        """
        Draw 'count' entries from memory.

        :param count: the amount of entries to draw
        :return: the drawn entries.
        """
        if self.valid_entries > 0:
            indices = np.random.randint(low=0, high=self.valid_entries, size=(count, 1))
        else:
            raise ValueError("You cannot sample from zero elements! Push elements to the memory first.")

        return self.memory[indices].reshape(count, self.entry_size)
