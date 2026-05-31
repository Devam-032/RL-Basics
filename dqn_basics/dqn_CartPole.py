import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

# ── Device ────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ── Environment exploration ────────────────────────────────
env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()

print(f"Initial state: {state}")
print(f"State shape  : {state.shape}")
print(f"Action space : {env.action_space}")
print(f"n actions    : {env.action_space.n}\n")

total_reward = 0
step_dash    = 0
for step in range(200):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    state         = next_state
    step_dash    += 1

    if terminated or truncated:
        print(f"Episode ended at step : {step_dash}")
        print(f"Total reward          : {total_reward}")
        state, info  = env.reset()
        total_reward = 0
        step_dash    = 0

env.close()

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
        x = self.layer3(x)              # no activation — Q-values can be negative
        return x

# Verify Component 1
torch.manual_seed(42)
network = QNetwork(state_size=4, action_size=2).to(device)
print(network)
print(f"\nTotal parameters: {sum(p.numel() for p in network.parameters()):,}")

fake_state = torch.randn(1, 4).to(device)
q_values   = network(fake_state)
print(f"\nFake state  : {fake_state}")
print(f"Q-values    : {q_values}")
print(f"Best action : {q_values.argmax().item()}\n")

# ══════════════════════════════════════════════════════════
# COMPONENT 2 — Replay Buffer
# ══════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state,
            int(action),    # force scalar integer — prevents shape (N,2) bug
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

# Verify Component 2
buffer = ReplayBuffer(capacity=50000)
for i in range(5):
    s  = np.random.randn(4)
    a  = np.random.randint(2)
    r  = 1.0
    s2 = np.random.randn(4)
    d  = False
    buffer.push(s, a, r, s2, d)

print(f"Buffer size : {len(buffer)}")
states, actions, rewards, next_states, dones = buffer.sample(3)
print(f"Sampled batch:")
print(f"  states shape      : {states.shape}")
print(f"  actions shape     : {actions.shape}")
print(f"  rewards shape     : {rewards.shape}")
print(f"  next_states shape : {next_states.shape}")
print(f"  dones shape       : {dones.shape}\n")

# ══════════════════════════════════════════════════════════
# COMPONENT 3 — DQN Agent
# ══════════════════════════════════════════════════════════
class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size  = state_size
        self.action_size = action_size
        self.device      = device

        # ── Two networks ──────────────────────────────
        self.q_net          = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)

        # Make target identical to q_net at start
        self.target_network.load_state_dict(self.q_net.state_dict())

        # Target never receives gradients — updated manually only
        for param in self.target_network.parameters():
            param.requires_grad = False

        # ── Optimizer ─────────────────────────────────
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.0001)

        # ── Replay buffer ─────────────────────────────
        self.buffer = ReplayBuffer(capacity=50000)

        # ── Hyperparameters ───────────────────────────
        self.gamma         = 0.99   # discount factor
        self.epsilon       = 1.0    # start fully random
        self.epsilon_min   = 0.05   # never below 1% random
        self.epsilon_decay = 0.997  # multiply per episode
        self.batch_size    = 64     # experiences per update
        self.update_step   = 75    # hard update every N steps
        self.steps_done    = 0      # global step counter

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)   # explore

        state_tensor = torch.tensor(state, dtype=torch.float32)\
                           .unsqueeze(0).to(self.device)  # (1,4)
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state_tensor)           # (1,2)
        self.q_net.train()

        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        # Sample random batch from buffer
        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        # Move to device
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # ── Target Q-values (frozen target network) ───
        with torch.no_grad():
            next_q     = self.target_network(next_states)  # (64,2)
            max_next_q = next_q.max(dim=1)[0]              # (64,)
            target_q   = rewards + self.gamma * max_next_q * (1 - dones)

        # ── Predicted Q-values (q_net) ────────────────
        all_q  = self.q_net(states)                                    # (64,2)
        pred_q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)      # (64,)

        # ── Loss and backprop ─────────────────────────
        loss = nn.MSELoss()(pred_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ── Hard update target every N steps ─────────
        self.steps_done += 1
        if self.steps_done % self.update_step == 0:
            self.target_network.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

# ── Verify Component 3 ────────────────────────────────────
print("=" * 50)
print("COMPONENT 3 — DQNAgent verification")
print("=" * 50)

# Test 1 — action selection
torch.manual_seed(42)
agent     = DQNAgent(state_size=4, action_size=2, device=device)
fake_state = np.random.randn(4)
action     = agent.select_action(fake_state)
print(f"\n[1] Action selection")
print(f"    Selected action : {action}")
print(f"    Epsilon         : {agent.epsilon}")

# Test 2 — buffer filling
for _ in range(200):
    s  = np.random.randn(4)
    a  = np.random.randint(2)
    r  = 1.0
    s2 = np.random.randn(4)
    d  = False
    agent.store(s, a, r, s2, d)

print(f"\n[2] Replay buffer")
print(f"    Buffer size     : {len(agent.buffer)}")

# Test 3 — one learn step
loss = agent.learn()
print(f"\n[3] Learning")
print(f"    Loss            : {loss:.6f}")

# Test 4 — epsilon decay
agent.decay_epsilon()
print(f"\n[4] Epsilon decay")
print(f"    Epsilon after   : {agent.epsilon:.6f}")

# Test 5 — target network mechanics
torch.manual_seed(42)
agent2 = DQNAgent(state_size=4, action_size=2, device=device)

# Fill buffer
for _ in range(100):
    s  = np.random.randn(4)
    a  = np.random.randint(2)
    r  = 1.0
    s2 = np.random.randn(4)
    d  = False
    agent2.store(s, a, r, s2, d)

# Capture initial weights
q_w_init = agent2.q_net.layer1.weight.data.clone().cpu()
t_w_init = agent2.target_network.layer1.weight.data.clone().cpu()
are_same = torch.allclose(q_w_init, t_w_init)

print(f"\n[5] Target network")
print(f"    Q == Target at init    : {are_same}")

# One learn step — Q changes, target stays frozen
agent2.learn()
q_w_after = agent2.q_net.layer1.weight.data.clone().cpu()
t_w_after = agent2.target_network.layer1.weight.data.clone().cpu()

q_changed = not torch.allclose(q_w_init, q_w_after)
t_frozen  = torch.allclose(t_w_init, t_w_after)
print(f"    Q-network changed      : {q_changed}")
print(f"    Target stayed frozen   : {t_frozen}")

# Force update at step 100
agent2.steps_done = 99
agent2.learn()
t_w_updated = agent2.target_network.layer1.weight.data.clone().cpu()
t_updated   = not torch.allclose(t_w_init, t_w_updated)
print(f"    Target updated at 100  : {t_updated}")
print("\n" + "=" * 50)
print("All components verified — ready for Component 4")
print("=" * 50)

def train_dqn(episodes = 700,render = False):
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1",render_mode=render_mode)

    torch.manual_seed(42)
    agent = DQNAgent(state_size=4,action_size=2,device=device)

    episode_rewards = []
    episode_losses = []
    best_reward = 0

    print(f"Training for {episodes} episodes on {device}")
    print(f"{'Episode':>8}  {'Reward':>8}  {'Avg100':>8}  "
          f"{'Loss':>10}  {'Epsilon':>8}")
    print("-" * 55)

    for episode in range (episodes):
        
        state,info = env.reset()
        done = False
        total_reward = 0
        losses = []

        while not done:
            action = agent.select_action(state)

            next_state,reward,terminated,truncated,info = env.step(action=action)
            done = terminated or truncated

            agent.store(state,action,reward,next_state,done)

            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward+=reward

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

        if avg100 >= 500:
            print(f"\nSolved at episode {episode+1}! "
                  f"Average reward: {avg100:.1f}")
            break

    env.close()
    return episode_rewards, episode_losses, agent

episode_rewards, episode_losses, trained_agent = train_dqn(
    episodes=2000,
    render=False
)
import matplotlib.pyplot as plt

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
                         np.ones(window)/window,
                         mode='valid')
    axes[0].plot(range(window, len(episode_rewards)+1), avg,
                 color='lime', linewidth=2,
                 label=f'Avg{window}')
    axes[0].axhline(y=475, color='red', linestyle='--',
                    linewidth=1.5, alpha=0.8, label='Solved (475)')
    axes[0].axhline(y=max(avg), color='yellow', linestyle=':',
                    linewidth=1, alpha=0.6,
                    label=f'Best avg: {max(avg):.0f}')
    axes[0].set_title('Episode Rewards', fontsize=13)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_ylim([0, 520])
    axes[0].legend(facecolor='#2a2a2a', labelcolor='white', fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # Plot 2 — Loss curve
    axes[1].plot(episodes, episode_losses,
                 color='#f39c12', linewidth=1, alpha=0.8)
    axes[1].set_title('Training Loss per Episode', fontsize=13)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('MSE Loss')
    axes[1].grid(True, alpha=0.2)

    # Plot 3 — Reward distribution
    axes[2].hist(episode_rewards, bins=30,
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[2].axvline(x=np.mean(episode_rewards), color='lime',
                    linewidth=2,
                    label=f'Mean: {np.mean(episode_rewards):.0f}')
    axes[2].axvline(x=475, color='red', linestyle='--',
                    linewidth=1.5, label='Solved: 475')
    axes[2].set_title('Reward Distribution', fontsize=13)
    axes[2].set_xlabel('Total Reward')
    axes[2].set_ylabel('Count')
    axes[2].legend(facecolor='#2a2a2a', labelcolor='white')
    axes[2].grid(True, alpha=0.2)

    plt.suptitle(
        f'DQN CartPole — Best avg100: {max(avg):.0f}  '
        f'Mean reward: {np.mean(episode_rewards):.0f}',
        fontsize=13, fontweight='bold', color='white'
    )
    plt.tight_layout()
    plt.savefig('dqn_training.png', dpi=130,
                bbox_inches='tight', facecolor='#1a1a1a')
    plt.show()
    print("Saved dqn_training.png")

plot_training(episode_rewards, episode_losses)