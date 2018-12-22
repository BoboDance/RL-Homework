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

    def plot_observations_cartpole(self):
        fig, ax = plt.subplots(3, 2)

        ax[0, 0].hist(self.memory[0:self.valid_entries, 0])
        ax[0, 0].set_title("x")
        ax[0, 1].hist(self.memory[0:self.valid_entries, 3])
        ax[0, 1].set_title("x_dot")

        ax[1, 0].hist(self.memory[0:self.valid_entries, 1])
        ax[1, 0].set_title("sin(theta)")
        ax[1, 1].hist(self.memory[0:self.valid_entries, 2])
        ax[1, 1].set_title("cos(theta)")
        ax[2, 0].hist(self.memory[0:self.valid_entries, 4])
        ax[2, 0].set_title("theta_dot")

        ax[2, 1].hist(self.memory[0:self.valid_entries, 5])
        ax[2, 1].set_title("action")

        fig.tight_layout()

        plt.show()
