import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from openai import OpenAI
from env import MinigridDoorKeyFullyObs
from model import CnnMinigridPolicy, ReplayBuffer
import time
import re

OLLAMA_MODEL = "llama3.2:3b"   
OLLAMA_BASE_URL = "http://localhost:11434/v1"

def grid_to_str(env):
    grid = env.unwrapped.grid
    width = env.unwrapped.width
    height = env.unwrapped.height
    agent_pos = env.unwrapped.agent_pos
    carrying = env.unwrapped.carrying

    has_key = carrying is not None and carrying.type == 'key'

    SYMBOLS = {
        'wall':   '▇',
        'door':   'D',
        'key':    'K',
        'goal':   'G',
        None:     ' ',
    }

    horizontal_border = '+' + '---+' * width
    rows = []

    for y in range(height):
        rows.append(horizontal_border)
        row_content = '|'
        for x in range(width):
            if (x, y) == tuple(agent_pos):
                row_content += f" { 'L' if has_key else 'A' } |"
            else:
                cell = grid.get(x, y)
                obj_type = cell.type if cell is not None else None
                
                if obj_type == 'door' and cell.is_open:
                    symbol = ' '
                else:
                    symbol = SYMBOLS.get(obj_type, ' ')
                    
                row_content += f" {symbol} |"
        rows.append(row_content)
    
    rows.append(horizontal_border)
    return '\n'.join(rows)

def hard_update(local_model, target_model):
    target_model.load_state_dict(local_model.state_dict())

def reward_llm(state, client, prompt, max_retries=8):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"CURRENT STATE:\n{state}"}
    ]

    for attempt in range(max_retries):
        try:
            chat_response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                max_tokens=64,       
                temperature=0.0,
                top_p=1.0,
                timeout=10.0,
            )
            raw_content = chat_response.choices[0].message.content.strip()

            numbers = re.findall(r'-?\.?\d+\.?\d*', raw_content)

            if numbers:
                risposta = numbers[-1]
                try:
                    val = float(risposta)
                    if val > 0 or val == -0.005:
                        return val
                    else:
                        print(f"[Parsing] Valore non conforme: '{val}' → reward = -0.005")
                        return -0.005
                except ValueError:
                    pass

            print(f"[Parsing] Nessun numero trovato nell'output dell'LLM → reward = -0.005")
            return -0.005

        except Exception as e:
            wait = min(2 ** attempt, 5)
            print(f"[Errore LLM] tentativo {attempt+1}/{max_retries}: {e} — attendo {wait}s")
            if attempt == max_retries - 1:
                return -0.005
            time.sleep(wait)

    return -0.005

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
    plt.title('Andamento Reward con LLM')
    plt.plot(r_vlm, color='blue', alpha=0.3, label='Reward Episodio')
    if len(r_vlm) > 50:
        means = np.convolve(r_vlm, np.ones(50)/50, mode='valid')
        plt.plot(np.arange(49, len(r_vlm)), means, color='red', label='Media Mobile 50')
    plt.xlabel('Episodi')
    plt.ylabel('Reward con LLM')
    plt.legend()

    plt.tight_layout()
    save_dir = "figure"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(os.path.join(save_dir, "ddqn-37.png"))
    print("\nGrafico finale salvato come 'figure/ddqn-37.png'")
    plt.show()

def train():
    with open("prompt2.txt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    client = OpenAI(
        api_key="ollama",
        base_url=OLLAMA_BASE_URL,
    )

    print("Warmup Ollama (caricamento modello in VRAM)...")
    try:
        client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Reply: -0.005"}],
            max_tokens=16,
            temperature=0.0,
        )
        print("Warmup completato.\n")
    except Exception as e:
        print(f"[Warmup fallito] {e} — assicurati che Ollama sia in esecuzione con: ollama serve")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training su dispositivo: {device}")

    GRID_SIZE = 8
    env = MinigridDoorKeyFullyObs(size=GRID_SIZE)

    num_actions = env.action_space.n
    state_space = env.observation_space.shape
    print(f"Azioni: {num_actions}, Spazio Osservazioni: {state_space}")

    num_episodes       = 600
    epsilon_ub         = 1.0
    epsilon_lb         = 0.05
    epsilon_decay      = 500_000
    buffer_size        = 300_000
    update_after       = 2_000
    minibatch_size     = 64
    train_every        = 2
    target_update_freq = 4_000
    gamma              = 0.99
    learning_rate      = 0.00005

    dqn = CnnMinigridPolicy(input_shape=state_space, num_actions=num_actions).to(device)
    dqn_target = CnnMinigridPolicy(input_shape=state_space, num_actions=num_actions).to(device)
    hard_update(dqn, dqn_target)

    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    huber_loss = torch.nn.SmoothL1Loss()

    buffer = ReplayBuffer(num_actions=num_actions, memory_len=buffer_size)
    success_buffer = ReplayBuffer(num_actions=num_actions, memory_len=buffer_size)

    timesteps = 0
    all_rewards = []
    state_rewards = []
    losses_history = []

    for episode in range(num_episodes):
        state = env.reset()[0]
        ret = 0
        ret_state = 0
        done = False
        episode_transitions = []
        ep_start = time.perf_counter()

        while not done:
            epsilon = max(epsilon_lb, epsilon_ub - timesteps / epsilon_decay)

            if np.random.random() < epsilon:
                action = np.random.randint(low=0, high=num_actions)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                net_out = dqn(state_tensor).detach().cpu().numpy()
                action = np.argmax(net_out)

            state_str = grid_to_str(env)
            print(state_str)
            reward = reward_llm(state_str, client, prompt)
            print(reward)

            next_state, state_reward, terminated, truncated, _ = env.step(action)
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
        state_rewards.append(ret_state)

        ep_time = time.perf_counter() - ep_start
        print(f"Episode: {episode} - REWARD BASE = {ret_state:.3f} - REWARD LLM = {ret:.3f} - Epsilon = {epsilon:.3f} - Durata = {ep_time:.1f}s")

    print("Addestramento completato!")

    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "37-8x8.pth")
    torch.save({'model_params': dqn.state_dict(), 'timesteps': timesteps}, save_path)
    print(f"Modello salvato in {save_path}")

    plot_reward(all_rewards, state_rewards)

if __name__ == "__main__":
    train()