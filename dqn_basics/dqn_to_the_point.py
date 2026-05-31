import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

# ── Device ────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ══════════════════════════════════════════════════════════
# COMPONENT 1 — Q-Network
# ══════════════════════════════════════════════════════════
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_size)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# ══════════════════════════════════════════════════════════
# COMPONENT 2 — Replay Buffer
# ══════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state,
            int(action),
            reward,
            next_state,
            done
        ))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states      = torch.tensor(np.array([e[0] for e in experiences]),
                                   dtype=torch.float32)
        actions     = torch.tensor(np.array([e[1] for e in experiences]),
                                   dtype=torch.long)
        rewards     = torch.tensor(np.array([e[2] for e in experiences]),
                                   dtype=torch.float32)
        next_states = torch.tensor(np.array([e[3] for e in experiences]),
                                   dtype=torch.float32)
        dones       = torch.tensor(np.array([e[4] for e in experiences]),
                                   dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ══════════════════════════════════════════════════════════
# COMPONENT 3 — DQN Agent
# ══════════════════════════════════════════════════════════
class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size  = state_size
        self.action_size = action_size
        self.device      = device

        # Two networks
        self.q_net          = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_net.state_dict())

        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.buffer    = ReplayBuffer(capacity=10000)

        # Hyperparameters
        self.gamma         = 0.99
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995
        self.batch_size    = 64
        self.update_step   = 100
        self.steps_done    = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        state_tensor = torch.tensor(state, dtype=torch.float32)\
                           .unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        self.q_net.train()
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Target Q-values
        with torch.no_grad():
            next_q     = self.target_network(next_states)
            max_next_q = next_q.max(dim=1)[0]
            target_q   = rewards + self.gamma * max_next_q * (1 - dones)

        # Predicted Q-values
        all_q  = self.q_net(states)
        pred_q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Loss and update
        loss = nn.MSELoss()(pred_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard update target
        self.steps_done += 1
        if self.steps_done % self.update_step == 0:
            self.target_network.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

# ══════════════════════════════════════════════════════════
# COMPONENT 4 — Training Loop
# ══════════════════════════════════════════════════════════
def train_dqn(episodes=500, render=False):
    render_mode = "human" if render else None
    env         = gym.make("CartPole-v1", render_mode=render_mode)

    torch.manual_seed(42)
    agent = DQNAgent(state_size=4, action_size=2, device=device)

    episode_rewards = []
    episode_losses  = []
    best_reward     = 0

    print(f"Training for {episodes} episodes on {device}")
    print(f"{'Episode':>8}  {'Reward':>8}  {'Avg100':>8}  "
          f"{'Loss':>10}  {'Epsilon':>8}")
    print("-" * 55)

    for episode in range(episodes):
        state, info  = env.reset()
        done         = False
        total_reward = 0
        losses       = []

        while not done:
            # 1. Select action
            action = agent.select_action(state)

            # 2. Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 3. Store experience
            agent.store(state, action, reward, next_state, done)

            # 4. Learn
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            # 5. Move to next state
            state         = next_state
            total_reward += reward

        # End of episode
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        avg_loss = np.mean(losses) if losses else 0.0
        episode_losses.append(avg_loss)

        avg100 = np.mean(episode_rewards[-100:])

        if total_reward > best_reward:
            best_reward = total_reward

        if (episode + 1) % 10 == 0:
            print(f"{episode+1:>8}  {total_reward:>8.0f}  {avg100:>8.1f}  "
                  f"{avg_loss:>10.4f}  {agent.epsilon:>8.4f}")

        # Solved threshold
        if avg100 >= 475:
            print(f"\nSolved at episode {episode+1}! "
                  f"Average reward: {avg100:.1f}")
            break

    env.close()
    return episode_rewards, episode_losses, agent

# ══════════════════════════════════════════════════════════
# COMPONENT 5 — Visualisation
# ══════════════════════════════════════════════════════════
def plot_training(episode_rewards, episode_losses):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor('#1a1a1a')

    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    episodes = range(1, len(episode_rewards) + 1)

    # Plot 1 — Reward curve
    axes[0].plot(episodes, episode_rewards,
                 color='steelblue', alpha=0.4,
                 linewidth=0.8, label='Raw reward')
    window = min(100, len(episode_rewards))
    avg    = np.convolve(episode_rewards,
                         np.ones(window) / window,
                         mode='valid')
    axes[0].plot(range(window, len(episode_rewards) + 1), avg,
                 color='lime', linewidth=2,
                 label=f'Avg {window}')
    axes[0].axhline(y=475, color='red', linestyle='--',
                    linewidth=1, alpha=0.7, label='Solved (475)')
    axes[0].set_title('Episode Rewards', fontsize=13)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_ylim([0, 520])
    axes[0].legend(facecolor='#2a2a2a', labelcolor='white')
    axes[0].grid(True, alpha=0.2)

    # Plot 2 — Loss curve
    axes[1].plot(episodes, episode_losses,
                 color='#f39c12', linewidth=1, alpha=0.7)
    axes[1].set_title('Training Loss', fontsize=13)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('MSE Loss')
    axes[1].grid(True, alpha=0.2)

    # Plot 3 — Reward distribution
    axes[2].hist(episode_rewards, bins=30,
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[2].axvline(x=np.mean(episode_rewards),
                    color='lime', linewidth=2,
                    label=f'Mean: {np.mean(episode_rewards):.0f}')
    axes[2].axvline(x=475, color='red', linestyle='--',
                    linewidth=1.5, label='Solved: 475')
    axes[2].set_title('Reward Distribution', fontsize=13)
    axes[2].set_xlabel('Total Reward')
    axes[2].set_ylabel('Count')
    axes[2].legend(facecolor='#2a2a2a', labelcolor='white')
    axes[2].grid(True, alpha=0.2)

    plt.suptitle('DQN Training — CartPole-v1',
                 fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig('dqn_training.png', dpi=130,
                bbox_inches='tight', facecolor='#1a1a1a')
    plt.show()
    print("Saved dqn_training.png")

def evaluate_agent(agent, episodes=10, render=True):
    """Watch the trained agent solve CartPole"""
    render_mode = "human" if render else None
    env         = gym.make("CartPole-v1", render_mode=render_mode)

    agent.epsilon = 0.0    # pure exploitation — no random actions
    rewards       = []

    print(f"\nEvaluating trained agent for {episodes} episodes...")
    print(f"{'Episode':>8}  {'Reward':>8}  {'Solved?':>8}")
    print("-" * 30)

    for episode in range(episodes):
        state, info  = env.reset()
        done         = False
        total_reward = 0

        while not done:
            action                                    = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done          = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)
        solved = "✓" if total_reward >= 475 else "✗"
        print(f"{episode+1:>8}  {total_reward:>8.0f}  {solved:>8}")

    env.close()
    print(f"\nMean reward : {np.mean(rewards):.1f}")
    print(f"Solved      : {sum(r >= 475 for r in rewards)}/{episodes} episodes")

# ══════════════════════════════════════════════════════════
# MAIN — Run everything
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    # Train
    episode_rewards, episode_losses, trained_agent = train_dqn(
        episodes=500,
        render=False       # set True to watch training (slower)
    )

    # Plot results
    plot_training(episode_rewards, episode_losses)

    # Summary
    print(f"\nTraining summary:")
    print(f"  Total episodes : {len(episode_rewards)}")
    print(f"  Best reward    : {max(episode_rewards):.0f}")
    print(f"  Mean reward    : {np.mean(episode_rewards):.1f}")
    print(f"  Last 100 avg   : {np.mean(episode_rewards[-100:]):.1f}")
    print(f"  Final epsilon  : {trained_agent.epsilon:.4f}")

    # Evaluate — watch the trained agent
    evaluate_agent(trained_agent, episodes=10, render=True)