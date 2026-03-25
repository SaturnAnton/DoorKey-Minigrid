import pickle
import gymnasium as gym
from minigrid.wrappers import *
import numpy as np
import time
import os

def load_file(filename):
    folder_path = os.path.join("..","data", "q_table")
    filepath = os.path.join(folder_path,filename)
    with open(filepath,"rb") as handle:
        qtable = pickle.load(handle)
    return qtable

def get_state_key(obs):
    img_flat = obs['image'][:, :, 0].flatten()  # Prende il primo strato della matrice e restituisce una lista monodimensionale
    direction = obs['direction']
    return str(list(img_flat) + [direction])

env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="human")
env = FullyObsWrapper(env)

obs, info = env.reset()
s = get_state_key(obs)
done = False

q_table = load_file("test.pkl")

while not done:
    a = np.argmax(q_table[s])
    
    obs, r, terminated, truncated, info = env.step(a)
    s = get_state_key(obs)
    
    done = terminated or truncated
    
    if r > 0:
        print("Vittoria!")
    
    time.sleep(0.1)

env.close()