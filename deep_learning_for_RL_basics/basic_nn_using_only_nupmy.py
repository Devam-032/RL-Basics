import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)

print(sigmoid(0))

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)           # shape: (4, 2)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

np.random.seed(42)

# W1: connects input layer (2 neurons) to hidden layer (4 neurons)
W1 = np.random.randn(2, 4) * 0.1    # shape: (2, 4)
b1 = np.zeros((1, 4))               # shape: (1, 4)  one bias per hidden neuron

# W2: connects hidden layer (4 neurons) to output layer (1 neuron)
W2 = np.random.randn(4, 1) * 0.1    # shape: (4, 1)
b2 = np.zeros((1, 1)) 

def forward_prop(X,W1,b1,W2,b2):
    Z1 = X@W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1@W2 + b2
    A2 = sigmoid(Z2)

    return Z1,A1,Z2,A2

Z1, A1, Z2, A2 = forward_prop(X, W1, b1, W2, b2)
print("Predictions before training:")
print(A2)


def compute_loss(A2,y):
    m = y.shape[0]
    loss = np.sum((A2-y)**2)/m
    return loss

loss = compute_loss(A2,y)
print(f"Loss before training: {loss:.4f}")

def backward_prop(X,y,Z1,A1,Z2,A2,W1,W2):
    m = y.shape[0]

    # How much did A2 contribute to the loss?
    dA2 = (2 / m) * (A2 - y) 

    # How much did Z2 contribute to the loss?
    dZ2 = dA2 * sigmoid_derivative(Z2)

    # How much did W2 and b2 contribute
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2,axis=0,keepdims=True)

    # Hidden Layer gradients
    # Propagate error back through W2
    dA1 = dZ2 @ W2.T

    # Chain Rule through sigmoid
    dZ1 = dA1*sigmoid_derivative(Z1)

    #How much did W1 and b1 contribute
    dW1 =  X.T @ dZ1
    db1 = np.sum(dZ1,axis=0,keepdims=True)

    return dW1,db1,dW2,db2

learning_rate = 0.05

print(f"{'Epoch':>6}  {'Loss':>8}")
print("-" * 20)

for epoch in range(1000000):
    #Forward pass - make predictions
    Z1,A1,Z2,A2 = forward_prop(X,W1,b1,W2,b2)

    #compute loss
    loss = compute_loss(A2,y)

    #Backward pass - compute gradients
    dW1, db1, dW2, db2 = backward_prop(X, y, Z1, A1, Z2, A2, W1, W2)

    #Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2   

    if epoch % 500 ==0:
        print(f"{epoch:>6}  {loss:>8.4f}")


print("-" * 20)
print(f"{'Final':>6}  {loss:>8.4f}")


print("\nFinal predictions after training:")
print(f"{'Input':>12}  {'Predicted':>10}  {'Target':>8}  {'Correct?':>8}")
print("-" * 48)

_, _, _, A2_final = forward_prop(X, W1, b1, W2, b2)

for i in range(4):
    inp = X[i].tolist()
    pred = A2_final[i][0]
    target = int(y[i][0])
    rounded = round(pred)
    correct = "✓" if rounded == target else "✗"
    print(f"{str(inp):>12}  {pred:>10.4f}  {target:>8}  {correct:>8}")