import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# print(f"Training samples : {len(train_dataset)}")
# print(f"Test samples     : {len(test_dataset)}")
# print(f"Training batches : {len(train_loader)}")
# print(f"Test batches     : {len(test_loader)}")

# # Peek at one batch
# images, labels = next(iter(train_loader))
# print(f"\nOne batch:")
# print(f"  images shape : {images.shape}")   # (64, 1, 28, 28)
# print(f"  labels shape : {labels.shape}")   # (64,)
# print(f"  pixel range  : {images.min():.3f} to {images.max():.3f}")

# fig, axes = plt.subplots(2, 8, figsize=(14, 4))
# axes = axes.flatten()

# for i in range(16):
#     img   = images[i].squeeze()        # remove channel dim: (1,28,28) → (28,28)
#     label = labels[i].item()
#     axes[i].imshow(img, cmap='gray')
#     axes[i].set_title(str(label), fontsize=12)
#     axes[i].axis('off')

# plt.suptitle('Sample MNIST images — what the network will learn from', y=1.02)
# plt.tight_layout()
# plt.savefig('mnist_samples.png', dpi=120)
# plt.show()
# print("Saved mnist_samples.png")

class MNISTnet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784,256)
        self.layer2 = nn.Linear(256,128)
        self.layer3 = nn.Linear(128,64)
        self.layer4 = nn.Linear(64,32)
        self.layer5 = nn.Linear(32,10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self,x):
        x = self.flatten(x)
        x = self.relu(self.layer1(x))
        x = self.dropout(x) 
        x = self.relu(self.layer2(x))
        x = self.dropout(x) 
        x = self.relu(self.layer3(x))
        x = self.dropout(x) 
        x = self.relu(self.layer4(x))
        x = self.dropout(x) 
        x = (self.layer5(x))
        return x
    
torch.manual_seed(42)
model = MNISTnet()
print(model)
print()

# ---Initial check---
# images, labels = next(iter(train_loader))
# print(f"Input shape  : {images.shape}")

# model.eval()
# with torch.no_grad():
#     output = model(images)

# print(f"Output shape : {output.shape}")    # (64, 10)
# print(f"\nLogits for first image:")
# print(f"  {output[0].numpy()}")
# print(f"\nTrue label for first image: {labels[0].item()}")
# print(f"Predicted class            : {output[0].argmax().item()}")
# print(f"Correct?                   : {output[0].argmax().item() == labels[0].item()}")

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

EPOCHS = 10

train_losses = []
train_accs = []

for epoch in range(EPOCHS):
    model.train()

    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    for batch_idx,(images,labels) in enumerate(train_loader):

        outputs = model(images)
        loss = loss_fn(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = outputs.argmax(dim=1)
        epoch_correct += (predicted == labels).sum().item()
        epoch_total   += len(labels)
        epoch_loss    += loss.item()
    
    avg_loss = epoch_loss/len(train_loader)
    avg_acc = (epoch_correct/epoch_total)*100
    train_losses.append(avg_loss)
    train_accs.append(avg_acc)

    print(f"Epoch {epoch+1}/{EPOCHS}  "
          f"Loss: {avg_loss:.4f}  "
          f"Accuracy: {avg_acc:.2f}%")

print("\nTraining complete.")

model.eval()

test_correct = 0
test_total   = 0
test_loss    = 0

# Collect wrong predictions
wrong_images  = []
wrong_labels  = []
wrong_preds   = []
confusion = torch.zeros(10, 10, dtype=torch.int)

with torch.no_grad():   # no gradient tracking needed — saves memory
    for images, labels in test_loader:
        outputs   = model(images)
        loss      = loss_fn(outputs, labels)

        predicted  = outputs.argmax(dim=1)
        test_correct += (predicted == labels).sum().item()
        test_total   += len(labels)
        test_loss    += loss.item()

        for t, p in zip(labels, predicted):
            confusion[t][p] += 1

        # Find wrong ones in this batch
        wrong_mask = predicted != labels
        wrong_images.append(images[wrong_mask])
        wrong_labels.append(labels[wrong_mask])
        wrong_preds.append(predicted[wrong_mask])

avg_test_loss = test_loss    / len(test_loader)
avg_test_acc  = test_correct / test_total * 100

print(f"Test Loss     : {avg_test_loss:.4f}")
print(f"Test Accuracy : {avg_test_acc:.2f}%")
print(f"Test Correct  : {test_correct} / {test_total}")

# Concatenate all wrong predictions
wrong_images = torch.cat(wrong_images)
wrong_labels = torch.cat(wrong_labels)
wrong_preds  = torch.cat(wrong_preds)

print(f"Total wrong: {len(wrong_images)} / 10000")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# ── Plot 1 — Training Loss ─────────────────────────────────
axes[0,0].plot(range(1, EPOCHS+1), train_losses,
               color='steelblue', linewidth=2, marker='o', markersize=8)
axes[0,0].set_title('Training Loss per Epoch', fontsize=13)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('CrossEntropy Loss')
axes[0,0].set_xticks(range(1, EPOCHS+1))
axes[0,0].grid(True, alpha=0.3)
for i, v in enumerate(train_losses):
    if i == 0 or i == len(train_losses)-1:
        axes[0,0].annotate(f'{v:.3f}',
                           (i+1, v),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center',
                           fontsize=10,
                           color='white',
                           fontweight='bold')

# ── Plot 2 — Training vs Test Accuracy ────────────────────
axes[0,1].plot(range(1, EPOCHS+1), train_accs,
               color='lime', linewidth=2, marker='o',
               markersize=8, label='Train accuracy')
axes[0,1].axhline(y=avg_test_acc,
                  color='red', linestyle='--',
                  linewidth=2,
                  label=f'Test accuracy ({avg_test_acc:.1f}%)')
axes[0,1].fill_between(range(1, EPOCHS+1),
                        train_accs, avg_test_acc,
                        alpha=0.1, color='red',
                        label='Generalisation gap')
axes[0,1].set_title('Train vs Test Accuracy', fontsize=13)
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Accuracy (%)')
axes[0,1].set_ylim([85, 100])
axes[0,1].set_xticks(range(1, EPOCHS+1))
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# ── Plot 3 — Confusion Matrix Heatmap ─────────────────────
im = axes[0,2].imshow(confusion.numpy(), cmap='Blues')
axes[0,2].set_title('Confusion Matrix', fontsize=13)
axes[0,2].set_xlabel('Predicted label')
axes[0,2].set_ylabel('True label')
axes[0,2].set_xticks(range(10))
axes[0,2].set_yticks(range(10))
# Annotate each cell with the count
for i in range(10):
    for j in range(10):
        val = confusion[i][j].item()
        color = 'white' if val > confusion.max().item() * 0.5 else 'black'
        axes[0,2].text(j, i, str(val),
                       ha='center', va='center',
                       fontsize=7, color=color)
plt.colorbar(im, ax=axes[0,2])

# ── Plot 4 — Per-class accuracy bar chart ─────────────────
class_acc = [confusion[i][i].item() /
             confusion[i].sum().item() * 100
             for i in range(10)]
colors = ['#2ecc71' if a >= 97 else
          '#f39c12' if a >= 94 else
          '#e74c3c' for a in class_acc]
bars = axes[1,0].bar(range(10), class_acc, color=colors, edgecolor='white')
axes[1,0].set_title('Per-class Accuracy', fontsize=13)
axes[1,0].set_xlabel('Digit class')
axes[1,0].set_ylabel('Accuracy (%)')
axes[1,0].set_xticks(range(10))
axes[1,0].set_ylim([85, 100])
axes[1,0].axhline(y=avg_test_acc, color='white',
                  linestyle='--', linewidth=1, alpha=0.5,
                  label=f'Overall ({avg_test_acc:.1f}%)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.2, axis='y')
for bar, acc in zip(bars, class_acc):
    axes[1,0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.1,
                   f'{acc:.1f}%',
                   ha='center', va='bottom', fontsize=8)

# ── Plot 5 — Wrong predictions gallery ────────────────────
axes[1,1].axis('off')
wrong_grid_ax = axes[1,1]
inner = fig.add_axes([0.36, 0.08, 0.28, 0.38])
inner.axis('off')
inner.set_title('Sample wrong predictions\n(True → Predicted)',
                fontsize=11, pad=10)

n_wrong_show = min(16, len(wrong_images))
grid_size    = 4

for idx in range(n_wrong_show):
    ax_inner = fig.add_axes([
        0.362 + (idx % grid_size) * 0.065,
        0.09  + (idx // grid_size) * 0.14,
        0.055, 0.12
    ])
    img = wrong_images[idx].squeeze()
    ax_inner.imshow(img, cmap='gray')
    ax_inner.set_title(
        f"{wrong_labels[idx].item()}→{wrong_preds[idx].item()}",
        fontsize=7, color='red', pad=2
    )
    ax_inner.axis('off')

# ── Plot 6 — Final summary stats ──────────────────────────
axes[1,2].axis('off')
summary_text = [
    ("MNIST FINAL RESULTS", 0.92, 14, 'white',  'bold'),
    ("─" * 32,              0.84, 9,  '#888888', 'normal'),
    (f"Test Accuracy   :  {avg_test_acc:.2f}%",
                            0.75, 12, '#2ecc71', 'bold'),
    (f"Test Loss       :  {avg_test_loss:.4f}",
                            0.66, 11, 'white',   'normal'),
    (f"Correct         :  {test_correct} / {test_total}",
                            0.57, 11, 'white',   'normal'),
    (f"Wrong           :  {test_total - test_correct} / {test_total}",
                            0.48, 11, '#e74c3c', 'normal'),
    ("─" * 32,              0.40, 9,  '#888888', 'normal'),
    (f"Best digit      :  {class_acc.index(max(class_acc))}  "
     f"({max(class_acc):.1f}%)",
                            0.31, 11, '#2ecc71', 'normal'),
    (f"Hardest digit   :  {class_acc.index(min(class_acc))}  "
     f"({min(class_acc):.1f}%)",
                            0.22, 11, '#e74c3c', 'normal'),
    ("─" * 32,              0.14, 9,  '#888888', 'normal'),
    ("Model: 784→256→128→64→16→10",
                            0.07, 9,  '#888888', 'normal'),
]
for text, y, size, color, weight in summary_text:
    axes[1,2].text(0.1, y, text,
                   transform=axes[1,2].transAxes,
                   fontsize=size, color=color,
                   fontweight=weight, va='center',
                   fontfamily='monospace')

plt.suptitle(f'MNIST Complete Results Dashboard',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('mnist_dashboard.png', dpi=130, bbox_inches='tight')
plt.show()
print("Saved mnist_dashboard.png")