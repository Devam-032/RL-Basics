import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
],dtype=torch.float32)

Y  = torch.tensor([
    [0],
    [1],
    [1],
    [0]
],dtype=torch.float32)

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,4,dtype=torch.float32)
        self.layer2 = nn.Linear(4,1,dtype=torch.float32)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.layer1(x))  # Z1 = xW1+b1, A1 = sigmoid(Z1)
        x = self.sigmoid(self.layer2(x))  # Z2 = A1W2+b2, A2 = sigmoid(Z2)
        return x
    
# Setup #

torch.manual_seed(42)
model     = MyNetwork()
loss_fn   = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop #
print(f"{'Epoch':>6}  {'Loss':>8}")
print("-" * 20)

for epoch in range(10000):
    
    pred = model(X)
    loss = loss_fn(pred,Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"{epoch:>6}  {loss.item():>8.4f}")

print("-" * 20)
print(f"{'Final':>6}  {loss.item():>8.4f}")

# Test #
print("\nFinal predictions:")
print(f"{'Input':>12}  {'Predicted':>10}  {'Target':>8}  {'Correct?':>8}")
print("-" * 48)

model.eval()
with torch.no_grad():
    predictions = model(X)

for i in range(4):
    inp    = X[i].tolist()
    pred   = predictions[i].item()
    target = int(Y[i].item())
    correct = "✓" if round(pred) == target else "✗"
    print(f"{str(inp):>12}  {pred:>10.4f}  {target:>8}  {correct:>8}")