import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)

X_np = np.linspace(-np.pi,np.pi,1000)
noise = np.random.normal(0,0.1,size = X_np.shape)
y_np = np.sin(X_np)+noise

X = torch.tensor(X_np,dtype = torch.float32).unsqueeze(1)
Y = torch.tensor(y_np,dtype = torch.float32).unsqueeze(1)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"X range: {X.min():.3f} to {X.max():.3f}")
print(f"Y range: {Y.min():.3f} to {Y.max():.3f}")

# # ── Plot raw data ─────────────────────────────────────────
# plt.figure(figsize=(10, 4))
# plt.scatter(X_np, y_np, s=1, alpha=0.5, color='steelblue', label='Noisy data')
# plt.plot(X_np, np.sin(X_np), color='red', linewidth=2, label='True sin(x)')
# plt.title('Sine wave — what the network needs to learn')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

class RegressionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1,16)
        self.layer2 = nn.Linear(16,16)
        self.layer3 = nn.Linear(16,16)
        self.layer4 = nn.Linear(16,16)
        self.layer5 = nn.Linear(16,1)
        self.model = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self,x):
        x = self.model(self.layer1(x))
        x = self.model(self.layer2(x))
        x = self.model(self.layer3(x))
        x = self.model(self.layer4(x))
        x = self.model(self.layer5(x))
        return x
    
model = RegressionNetwork()

# # Untrained prediction — should be garbage
# model.eval()
# with torch.no_grad():
#     y_untrained = model(X)

# plt.figure(figsize=(10, 4))
# plt.scatter(X_np, y_np, s=1, alpha=0.3, color='steelblue', label='Noisy data')
# plt.plot(X_np, np.sin(X_np),              color='red',    linewidth=2, label='True sin(x)')
# plt.plot(X_np, y_untrained.numpy(),       color='orange', linewidth=2, label='Untrained network')
# plt.title('Before training — network has no idea')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('sine_untrained.png', dpi=120)
# plt.show()

loss_fn   = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training

EPOCHS = 5000
loss_hist = []

print(f"{'Epoch':>6}  {'Loss':>10}")
print("-" * 22)

for epoch in range(EPOCHS):
    model.train()

    pred = model(X)
    loss = loss_fn(pred,Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_hist.append(loss.item())

    if epoch%500==0:
        print(f"{epoch:>6} {loss.item():>10.6f}")

print("-" * 22)
print(f"{'Final':>6}  {loss.item():>10.6f}")

plt.figure(figsize=(10, 4))
plt.plot(loss_hist, color='steelblue', linewidth=1)
plt.title('Training loss over time')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log')        # log scale — makes the early drop visible
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sine_loss_curve.png', dpi=120)
plt.show()

model.eval()
with torch.no_grad():
    y_predicted = model(X)

# Convert back to numpy for plotting
y_predicted_np = y_predicted.numpy()

plt.figure(figsize=(10, 4))
plt.scatter(X_np, y_np,
            s=1, alpha=0.3,
            color='steelblue',
            label='Noisy data')

plt.figure(figsize=(10, 4))
plt.scatter(X_np, y_np,
            s=1, alpha=0.3,
            color='steelblue',
            label='Noisy data')

plt.plot(X_np, np.sin(X_np),
         color='red',
         linewidth=2,
         label='True sin(x)')

plt.plot(X_np, y_predicted_np,
         color='lime',
         linewidth=2,
         label='Network prediction')

plt.title('After training — network learned sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sine_final.png', dpi=120)
plt.show()