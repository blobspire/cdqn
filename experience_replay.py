# Define memory for Experience Replay

from collections import deque
import random

class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    # Add experience to memory
    # transition = (state, action, reward, next_state, terminated)
    def append(self, transition):
        self.memory.append(transition)

    # Sample from memory of batch size sample_size
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)