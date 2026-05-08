import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from monorepo import GroqLLM, load_api_keys, GROQ_MULTIMODAL_MODEL_ID
import io
import time

from env import MinigridDoorKeyFullyObs
from model import CnnMinigridPolicy, ReplayBuffer

def hard_update(local_model, target_model):
    target_model.load_state_dict(local_model.state_dict())


def reward_vlm(img_array, client, prompt, max_retries=8):
    
    time.sleep(25)
    buffer = io.BytesIO()
    Image.fromarray(img_array.astype(np.uint8)).save(buffer, format="JPEG")
    buffer.seek(0)
    pil_image = Image.open(buffer)

    for attempt in range(max_retries):
        try:
            risposta = client.ask(prompt=prompt, images=[pil_image])
            try:
                return float(risposta.strip())
            except ValueError:
                print(f"[Parsing] Risposta non numerica: '{risposta}' — reward = -0.005")
                return -0.005

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower() or "too many" in str(e).lower():
                wait = (2 ** attempt) + 60
                print(f"[Rate limit Groq] Attendo {wait:.1f}s (tentativo {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"[Errore VLM] {e} — reward = -0.005")
                return -0.005

    print("[Max retries] Groq non disponibile, reward = -0.005")
    return -0.005

def final_plot(rewards, losses):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.title('Andamento Reward Totale')
    plt.plot(rewards, color='blue', alpha=0.3, label='Reward Episodio')
    if len(rewards) > 50:
        means = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(np.arange(49, len(rewards)), means, color='red', label='Media Mobile 50')
    plt.xlabel('Episodi')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(122)
    plt.title('Andamento Loss')
    if len(losses) > 0:
        plt.plot(losses, color='orange')
    plt.xlabel('Step di ottimizzazione')
    plt.ylabel('Loss')

    plt.tight_layout()
    save_dir = "figure"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(os.path.join(save_dir, "ddqn-23.png"))
    print("\nGrafico finale salvato come 'figure/ddqn-23.png'")
    plt.show()

#fai due grafici dei reward :un reward con quello vecchio e uno sulla vlm
def plot_reward(r_r, r_vlm):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.title('Andamento Reward Totale')
    plt.plot(r_r, color='blue', alpha=0.3, label='Reward Episodio')
    if len(r_r) > 50:
        means = np.convolve(r_r, np.ones(50)/50, mode='valid')
        plt.plot(np.arange(49, len(r_r)), means, color='red', label='Media Mobile 50')
    plt.xlabel('Episodi')
    plt.ylabel('Reward di base')
    plt.legend()

    plt.subplot(122)
    plt.title('Andamento Reward con VLM')
    plt.plot(r_vlm, color='blue', alpha=0.3, label='Reward Episodio')
    if len(r_vlm) > 50:
        means = np.convolve(r_vlm, np.ones(50)/50, mode='valid')
        plt.plot(np.arange(49, len(r_vlm)), means, color='red', label='Media Mobile 50')
    plt.xlabel('Episodi')
    plt.ylabel('Reward con VLM')
    plt.legend()

    plt.tight_layout()
    save_dir = "figure"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(os.path.join(save_dir, "ddqn-27.png"))
    print("\nGrafico finale salvato come 'figure/ddqn-27.png'")
    plt.show()

def train():
    load_api_keys()
    with open("prompt1.txt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    
    client = GroqLLM(model_id=GROQ_MULTIMODAL_MODEL_ID)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training su dispositivo: {device}")

    GRID_SIZE = 8
    env = MinigridDoorKeyFullyObs(size=GRID_SIZE)
    
    num_actions = env.action_space.n
    state_space = env.observation_space.shape
    print(f"Azioni: {num_actions}, Spazio Osservazioni: {state_space}")

    num_episodes       = 10   
    buffer_size        = 200000   
    epsilon_ub         = 1.0
    epsilon_lb         = 0.05
    epsilon_decay      = 1_500_000
    minibatch_size     = 128
    gamma              = 0.99
    learning_rate      = 0.00005
    update_after       = 2000     
    train_every        = 4
    target_update_freq = 2000     

    dqn = CnnMinigridPolicy(input_shape=state_space, num_actions=num_actions).to(device)
    dqn_target = CnnMinigridPolicy(input_shape=state_space, num_actions=num_actions).to(device)
    hard_update(dqn, dqn_target)
    
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    huber_loss = torch.nn.SmoothL1Loss()
    
    #cambiata la dimensione
    buffer = ReplayBuffer(num_actions=num_actions, memory_len=buffer_size)
    success_buffer = ReplayBuffer(num_actions=num_actions, memory_len=buffer_size)

    timesteps = 0
    returns_50 = []
    all_rewards = []
    state_rewards = []   
    losses_history = [] 

    for episode in range(num_episodes):
        state = env.reset()[0]
        ret = 0
        ret_state = 0
        done = False
        episode_transitions = []
        check = 0

        while not done:
            epsilon = max(epsilon_lb, epsilon_ub - timesteps / epsilon_decay)
            
            if np.random.random() < epsilon:
                action = np.random.randint(low=0, high=num_actions)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                net_out = dqn(state_tensor).detach().cpu().numpy()
                action = np.argmax(net_out)

            img = env.render()
            reward = reward_vlm(img,client,prompt)
            print(check)
            check += 1
            next_state, state_reward , terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ret += reward
            ret_state += state_reward

            buffer.add(state, action, reward, next_state, done)
            episode_transitions.append((state, action, reward, next_state, done))
            
            state = next_state
            timesteps += 1

            if timesteps % train_every == 0 and buffer.length() > minibatch_size and buffer.length() > update_after:
                optimizer.zero_grad()

                states_mb, a_mb, reward_mb, next_states_mb, done_mb = buffer.sample_batch(device, minibatch_size)
                
                if success_buffer.length() > 8:
                    s_states, s_a, s_reward, s_next, s_done = success_buffer.sample_batch(device, 8)
                    states_mb = np.concatenate([states_mb, s_states], axis=0)
                    a_mb = torch.cat([a_mb, s_a], dim=0)
                    reward_mb = torch.cat([reward_mb, s_reward], dim=0)
                    next_states_mb = np.concatenate([next_states_mb, s_next], axis=0)
                    done_mb = torch.cat([done_mb, s_done], dim=0)

                states_t = torch.tensor(states_mb, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states_mb, dtype=torch.float32, device=device)

                q_values = dqn(states_t)

                with torch.no_grad():
                    q_next_online = dqn(next_states_t)
                    best_actions = torch.argmax(q_next_online, dim=1)
                    q_next_target = dqn_target(next_states_t)
                    q_next_value = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                    targets = reward_mb + gamma * q_next_value * (1 - done_mb)

                predictions = torch.sum(q_values * a_mb, dim=1)
                loss = huber_loss(predictions, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dqn.parameters(), 10)
                optimizer.step()
                
                losses_history.append(loss.item())

            if timesteps % target_update_freq == 0:
                hard_update(dqn, dqn_target)

        if ret > 0:
            for s, a_t, r_t, ns, d in episode_transitions:
                success_buffer.add(s, a_t, r_t, ns, d)

        all_rewards.append(ret)
        state_rewards.append(state_reward)
        returns_50.append(ret)
        if len(returns_50) > 50:
            returns_50.pop(0)

        if episode % 50 == 0:
            avg_return = np.mean(returns_50) if len(returns_50) > 0 else 0
            print(f"Episodio {episode}\tMedia Ritorno (ultimi 50): {avg_return:.2f}\tEpsilon: {epsilon:.3f}")

        print(episode)

    print("Addestramento completato!")
    
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "27-8x8.pth")
    torch.save({'model_params': dqn.state_dict(), 'timesteps': timesteps}, save_path)
    print(f"Modello salvato in {save_path}")

    plot_reward(all_rewards,state_rewards)

if __name__ == "__main__":
    train()
