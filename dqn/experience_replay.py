from collections import deque
import random


class ReplayMemory:
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    #aggiungere le esperienze alla memoria
    def append(self, transition):
        self.memory.append(transition)

    #campionba casualmente la memoria e restituisce la
    #dimensione del batch che abbiamo specificato
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    #ritorna la lunghezza della memoria
    def __len__(self):
        return len(self.memory)