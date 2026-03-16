import pickle
import gymnasium as gym
from minigrid.wrappers import *
import numpy as np
import time
import os

def load_file(filename):
    filepath = os.path.join("q_table",filename)
    with open(filepath,"rb") as handle:
        qtable = pickle.load(handle)
    return qtable

def get_state_key(obs):
    img_flat = obs['image'][:, :, 0].flatten()  # Prende il primo strato della matrice e restituisce una lista monodimensionale
    direction = obs['direction']
    return str(list(img_flat) + [direction])

env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")

obs, info = env.reset()
s = get_state_key(obs)
done = False

q_table = load_file("qtable10.pkl")

while not done:
    a = np.argmax(q_table[s])
    
    obs, r, terminated, truncated, info = env.step(a)
    s = get_state_key(obs)
    
    done = terminated or truncated
    
    if r > 0:
        print("Vittoria!")
    
    time.sleep(0.1)

env.close()