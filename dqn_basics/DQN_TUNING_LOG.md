# DQN CartPole — Hyperparameter Tuning Log

## Environment
- **Task:** CartPole-v1
- **Solved threshold:** avg100 ≥ 475
- **Device:** CUDA (NVIDIA GPU)

---

## Baseline Architecture (fixed across all runs)
```
QNetwork: 4 → 64 → 64 → 2
Activation: ReLU (hidden), None (output)
Optimizer: Adam
Buffer capacity: varies (see runs)
Batch size: varies (see runs)
Gamma: 0.99
Epsilon start: 1.0
```

---

## Master Run Table

| Run | Algorithm | lr     | eps_decay | eps_min | target_step | buffer | batch | other            | avg100 | mean | solved |
|-----|-----------|--------|-----------|---------|-------------|--------|-------|------------------|--------|------|--------|
| 1   | Vanilla   | 0.001  | 0.995     | 0.01    | 100         | 10k    | 64    | baseline         | 311    | ~150 | ❌     |
| 2   | Vanilla   | 0.001  | 0.997     | 0.01    | 50          | 10k    | 64    | 2 params ❌      | 446    | ~250 | ❌     |
| 3   | Vanilla   | 0.001  | 0.995     | 0.01    | 50          | 10k    | 64    | clip=1.0         | 341    | ~180 | ❌     |
| 4   | Vanilla   | 0.001  | 0.995     | 0.01    | 75          | 10k    | 64    | SmoothL1Loss     | 124    | ~40  | ❌     |
| 5   | Vanilla   | 0.003  | 0.995     | 0.01    | 75          | 10k    | 64    | SmoothL1Loss     | 107    | ~35  | ❌     |
| 6   | Vanilla   | 0.001  | 0.995     | 0.01    | 75          | 10k    | 64    | no clip, no norm | 448    | ~220 | ❌     |
| 6b  | Vanilla   | 0.001  | 0.995     | 0.01    | 75          | 10k    | 64    | norm ON          | 283    | ~120 | ❌     |
| 7   | Vanilla   | 0.001  | 0.997     | 0.01    | 75          | 10k    | 64    | 2000 eps         | 459    | ~245 | ❌     |
| 8   | Vanilla   | 0.001  | 0.997     | 0.01    | 75          | 50k    | 64    | buffer 50k       | 462    | ~268 | ❌     |
| 9   | Vanilla   | 0.001  | 0.997     | 0.05    | 75          | 50k    | 64    | eps_min=0.05     | 413    | ~220 | ❌     |
| 10  | Vanilla   | 0.001  | 0.997     | 0.05    | 50          | 50k    | 64    | target_step=50   | 448    | ~253 | ❌     |
| 11  | Vanilla   | 0.0005 | 0.997     | 0.05    | 50          | 50k    | 64    | lr=0.0005        | 446    | ~285 | ❌     |
| 12  | Vanilla   | 0.0005 | 0.997     | 0.05    | 50          | 50k    | 128   | batch=128        | 447    | ~254 | ❌     |
| 13  | Vanilla   | 0.0005 | 0.997     | 0.05    | 50          | 50k    | 64    | adaptive eps     | 310    | ~145 | ❌     |
| 14  | Vanilla   | 0.0005 | 0.997     | 0.05    | 75          | 50k    | 64    | best-of-all      | 442    | 306  | ❌     |
| 15  | DDQN      | 0.0005 | 0.997     | 0.05    | 75          | 50k    | 64    | DDQN target      | 401    | 272  | ❌     |
| 16  | DDQN      | 0.001  | 0.997     | 0.05    | 75          | 50k    | 64    | DDQN + lr up     | 278    | 192  | ❌     |
| 17  | DDQN      | 0.0001 | 0.997     | 0.05    | 75          | 50k    | 64    | DDQN + lr=0.0001 | 500    | 302  | ✅     |
| 18  | Vanilla   | 0.0001 | 0.997     | 0.05    | 75          | 50k    | 64    | vanilla+lr=0.0001| 500    | 283  | ✅     |

---

## Key Findings Per Run

**Run 1 — Baseline:**
MSELoss, default config. First perfect episode at ep~280.
Classic instability dip after ep 320. Loss grew 1→40.
Root cause identified: MSELoss + high lr → Q-value explosion.

**Run 2 — Two params changed (violated one-param rule):**
Best result so far at the time (446). Proved epsilon_decay and
target_step both matter but couldn't isolate which helped.
Lesson: never change two params at once.

**Run 3 — Grad clipping max_norm=1.0:**
Clipping too tight — strangled gradients. Loss spikes still appeared.
Proved grad clipping alone cannot fix Q-value divergence.

**Run 4 — SmoothL1Loss:**
Loss spikes eliminated (max=5). But learning very slow.
avg100 only reached 124. Huber Loss too conservative for CartPole speed.

**Run 5 — SmoothL1Loss + lr=0.003:**
Higher lr with Huber — overshot, loss collapsed to 0.
Worse than Run 4. Huber has narrow stable lr range.

**Run 6 — MSELoss, no clipping, no normalisation:**
Best config base established. Fast learning, some instability.
No normalisation needed — CartPole state already well-scaled.

**Run 6b — Normalisation ON:**
Performance dropped 448→283. CartPole state doesn't need normalisation.
Rule confirmed: normalise only when input scales are very different.

**Run 7 — Slower epsilon decay + 2000 episodes:**
459 avg100. Agent reached near-solved multiple times then collapsed.
Periodic collapses tied to epsilon approaching minimum.

**Run 8 — Buffer 50k:**
Marginal improvement 459→462. Hypothesis A (buffer overwrite) rejected.
Collapses still occurred at same intervals.

**Run 9 — epsilon_min=0.05:**
Catastrophic collapses eliminated. But peak dropped 462→413.
Core tradeoff confirmed: stability vs performance.
Hypothesis B (epsilon exhaustion) confirmed as collapse cause.

**Run 10 — target_step=50:**
Performance recovered to 448. But instability returned, loss spiked to 840.
More frequent target updates accelerate both learning AND divergence.

**Run 11 — lr=0.0005:**
Best mean reward so far (285). Loss spikes halved (840→500).
Recovery after collapse faster. Peak unchanged at 446.

**Run 12 — batch=128:**
Slightly worse than 64 across all metrics. Slower early learning,
one extra collapse. Batch=64 better for CartPole scale.

**Run 13 — Adaptive epsilon:**
Worst result (310). Trigger too sensitive — fired on normal noise.
Created positive feedback loop of re-exploration. Agent never consolidated.
Concept valid but parameters need: longer window, smaller bump, higher threshold.

**Run 14 — Best-of-everything synthesis:**
Best mean reward (306) across all vanilla runs. Calmest loss curve (max=150).
Only 1 shallow collapse. New failure mode: gradual drift instead of sharp collapse.
True vanilla DQN ceiling established.

**Run 15 — Double DQN, lr=0.0005:**
Worse than vanilla Run 14. avg100=401. Loss calmer (max=90) confirming
overestimation IS being reduced, but performance dropped.
Root cause: lr=0.0005 tuned for vanilla's overestimated targets.
Double DQN's accurate targets make same lr too conservative.

**Run 16 — Double DQN, lr=0.001:**
Worse again (278). Higher lr caused premature convergence to wrong local optimum.
Loss stable after ep500 but reward plateaued at ~270 and couldn't escape.

**Run 17 — Double DQN, lr=0.0001:** ✅ SOLVED
Perfect monotonic reward climb. Solved at ~850 episodes.
lr=0.0001 + accurate DDQN targets = smooth convergence, zero collapses.

**Run 18 — Vanilla DQN, lr=0.0001:** ✅ SOLVED
Vanilla also solved at ~875 episodes. Mean=283 vs DDQN mean=302.
Proved: lr=0.0001 is the key insight, not just the architecture.

---

## Critical Learnings

**1. Learning rate was the root cause all along**
```
Runs 1–16:  lr ≥ 0.0005  →  never solved (16 runs)
Runs 17–18: lr = 0.0001  →  both solved

At lr=0.0001 Q-value overestimation accumulates so slowly
it never triggers catastrophic divergence. Weight updates are
tiny enough that even vanilla DQN's biased targets don't explode.
```

**2. Double DQN is a genuine improvement but not a requirement**
```
Both algorithms solved with lr=0.0001.
DDQN advantages: +19 mean reward, 25 eps faster, smoother loss.
Vanilla also solves: just slightly slower and noisier.
DDQN benefit scales with action space — more impactful on Atari (18 actions).
```

**3. Normalisation — know when to use it**
```
MNIST pixels (0–255): normalisation essential
CartPole state (-4.8 to 4.8): normalisation hurts
Rule: normalise when input scales differ significantly
```

**4. epsilon_min tradeoff**
```
0.01: high peak, violent collapses
0.05: stable floor, lower peak
0.05 combined with lr=0.0001 gave best of both
```

**5. Adaptive epsilon — concept valid, params were wrong**
```
Too sensitive threshold (50 pts over 10 eps) = fires on noise
Better: 100 pts over 50 eps, bump +0.05 only
```

**6. Grad clipping — max_norm matters**
```
1.0:  too tight, kills gradients
10.0: not tight enough for MSELoss
None: best for this network size with lr=0.0001
```

**7. Loss curve is the primary diagnostic tool**
```
Smooth rise then plateau   → healthy ✓
Spike then catastrophic    → lr too high or overestimation
Flat near zero from start  → lr too low or clip too tight
Rise then collapse to 0    → overshot minimum
Gradual drift down         → slow overestimation accumulation
```

---

## Final Solved Configuration

```python
# Works for BOTH Vanilla DQN and Double DQN

self.optimizer     = optim.Adam(self.q_net.parameters(), lr=0.0001)
self.buffer        = ReplayBuffer(capacity=50000)
self.gamma         = 0.99
self.epsilon       = 1.0
self.epsilon_min   = 0.05
self.epsilon_decay = 0.997
self.batch_size    = 64
self.update_step   = 75

# No grad clipping
# No normalisation
# MSELoss
# 2000 episodes max (solved ~850–875)
```

---

## Parameter Sensitivity Summary

| Parameter     | Too low effect        | Too high effect       | Best value  |
|---------------|----------------------|-----------------------|-------------|
| Learning rate | Never converges      | Q-value explosion     | **0.0001**  |
| Epsilon decay | Never exploits       | Buffer loses diversity| 0.997       |
| Epsilon min   | Violent collapses    | Low peak performance  | 0.05        |
| Target update | Target too unstable  | Target too stale      | 75 steps    |
| Buffer size   | Low diversity        | Memory cost (marginal)| 50k         |
| Batch size    | Noisy gradients      | Slower learning       | 64          |
| Grad clip     | Kills gradients      | No protection         | None        |

