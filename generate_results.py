"""
Generate all result plots and save them to results/.
Run from the micrograd_plus directory:
    python generate_results.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from micrograd import Tensor
from micrograd.nn import Sequential, Linear, ReLU, Sigmoid, Dropout
from micrograd.nn.loss import BCELoss, MSELoss, CrossEntropyLoss
from micrograd.optim import Adam, SGD

os.makedirs("results", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "figure.dpi":       150,
})
ACCENT   = "#58a6ff"
ACCENT2  = "#3fb950"
ACCENT3  = "#f78166"
ACCENT4  = "#d2a8ff"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  XOR — loss curve + decision boundary
# ─────────────────────────────────────────────────────────────────────────────
print("Generating XOR results ...")
np.random.seed(42)

X_xor = Tensor([[0,0],[0,1],[1,0],[1,1]], requires_grad=False)
y_xor = Tensor([[0],[1],[1],[0]], requires_grad=False)

model_xor = Sequential(Linear(2, 16), ReLU(), Linear(16, 8), ReLU(), Linear(8, 1), Sigmoid())
opt_xor   = Adam(list(model_xor.parameters()), lr=0.01)
crit_xor  = BCELoss()

xor_losses = []
EPOCHS_XOR = 2000
for epoch in range(EPOCHS_XOR):
    pred = model_xor(X_xor)
    loss = crit_xor(pred, y_xor)
    opt_xor.zero_grad()
    loss.backward()
    opt_xor.step()
    xor_losses.append(loss.item())

# ── figure: 1 row, 2 cols ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("XOR Problem — MLP (2 → 16 → 8 → 1)", color="#c9d1d9", fontsize=14, fontweight="bold")

# Loss curve
ax = axes[0]
ax.plot(xor_losses, color=ACCENT, linewidth=1.5, label="BCE Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss", color="#c9d1d9")
ax.grid(True)
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
ax.text(0.97, 0.97, f"Final loss: {xor_losses[-1]:.5f}",
        transform=ax.transAxes, ha='right', va='top',
        fontsize=10, color=ACCENT2,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#30363d'))

# Decision boundary
ax = axes[1]
h = 0.01
xx, yy = np.meshgrid(np.arange(-0.3, 1.3, h), np.arange(-0.3, 1.3, h))
grid   = Tensor(np.c_[xx.ravel(), yy.ravel()])
model_xor.eval()
Z = model_xor(grid).data.reshape(xx.shape)
model_xor.train()

ax.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.85)
ax.contour(xx,  yy, Z, levels=[0.5], colors='white', linewidths=1.2, linestyles='--')

colors  = [ACCENT3, ACCENT2]
labels  = ['Class 0', 'Class 1']
targets = [0, 1, 1, 0]
pts     = [[0,0],[0,1],[1,0],[1,1]]
for pt, t in zip(pts, targets):
    ax.scatter(pt[0], pt[1], c=colors[t], s=250, zorder=5,
               edgecolors='white', linewidths=1.5)
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_title("Decision Boundary", color="#c9d1d9")
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=ACCENT3, label='Class 0 (y=0)'),
                   Patch(facecolor=ACCENT2, label='Class 1 (y=1)')]
ax.legend(handles=legend_elements, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

plt.tight_layout()
plt.savefig("results/xor_results.png", bbox_inches='tight')
plt.close()
print("  Saved results/xor_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Linear Regression
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Regression results ...")
np.random.seed(42)

n = 200
X_np = np.random.uniform(-3, 3, (n, 1))
y_np = 2.0 * X_np + 1.0 + np.random.randn(n, 1) * 0.5

split = 160
X_train_t = Tensor(X_np[:split])
y_train_t = Tensor(y_np[:split])
X_test_t  = Tensor(X_np[split:])
y_test_t  = Tensor(y_np[split:])

model_reg  = Linear(1, 1)
opt_reg    = SGD(list(model_reg.parameters()), lr=0.01, momentum=0.9)
crit_reg   = MSELoss()

reg_train_losses, reg_test_losses = [], []
EPOCHS_REG = 500
for epoch in range(EPOCHS_REG):
    pred = model_reg(X_train_t)
    loss = crit_reg(pred, y_train_t)
    opt_reg.zero_grad()
    loss.backward()
    opt_reg.step()
    reg_train_losses.append(loss.item())
    if (epoch + 1) % 5 == 0:
        model_reg.eval()
        tl = crit_reg(model_reg(X_test_t), y_test_t).item()
        reg_test_losses.append(tl)
        model_reg.train()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Linear Regression: y = 2x + 1 + noise", color="#c9d1d9", fontsize=14, fontweight="bold")

# Loss curves
ax = axes[0]
ax.plot(reg_train_losses, color=ACCENT,  linewidth=1.5, label="Train MSE", alpha=0.9)
ax.plot(range(4, EPOCHS_REG, 5), reg_test_losses, color=ACCENT2, linewidth=1.5,
        label="Test MSE", linestyle='--')
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training & Test Loss", color="#c9d1d9")
ax.grid(True)
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

# Regression fit
ax = axes[1]
model_reg.eval()
w = model_reg.weight.data[0, 0]
b = model_reg.bias.data[0]

ax.scatter(X_np[:split], y_np[:split], color=ACCENT,  alpha=0.4, s=20, label="Train data")
ax.scatter(X_np[split:], y_np[split:], color=ACCENT4, alpha=0.7, s=30, label="Test data")

x_line = np.linspace(-3.2, 3.2, 200).reshape(-1, 1)
y_pred_line = w * x_line + b
y_true_line = 2.0 * x_line + 1.0

ax.plot(x_line, y_pred_line, color=ACCENT2,  linewidth=2.5, label=f"Learned: {w:.2f}x + {b:.2f}")
ax.plot(x_line, y_true_line, color=ACCENT3, linewidth=1.5, linestyle='--', label="True: 2.00x + 1.00")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Regression Fit", color="#c9d1d9")
ax.grid(True)
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)

y_test_pred = model_reg(X_test_t).data
ss_res = np.sum((y_np[split:] - y_test_pred) ** 2)
ss_tot = np.sum((y_np[split:] - y_np[split:].mean()) ** 2)
r2 = 1 - ss_res / ss_tot
ax.text(0.03, 0.97, f"R² = {r2:.4f}", transform=ax.transAxes,
        ha='left', va='top', fontsize=11, color=ACCENT2,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#30363d'))

plt.tight_layout()
plt.savefig("results/regression_results.png", bbox_inches='tight')
plt.close()
print("  Saved results/regression_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MNIST-like synthetic benchmark (illustrative curves)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating MNIST benchmark results (synthetic data) ...")
np.random.seed(0)

N_TRAIN, N_TEST, D, C = 4000, 800, 64, 10
X_syn   = np.random.randn(N_TRAIN + N_TEST, D).astype(np.float64)
y_syn   = np.random.randint(0, C, N_TRAIN + N_TEST)
X_tr, y_tr = X_syn[:N_TRAIN], y_syn[:N_TRAIN]
X_te, y_te = X_syn[N_TRAIN:], y_syn[N_TRAIN:]

model_mlp = Sequential(
    Linear(D, 128), ReLU(), Dropout(0.2),
    Linear(128, 64), ReLU(),
    Linear(64, C)
)
opt_mlp  = Adam(list(model_mlp.parameters()), lr=0.001)
crit_mlp = CrossEntropyLoss()

EPOCHS_MLP = 30
BATCH      = 128
mlp_train_loss, mlp_test_acc = [], []

for epoch in range(EPOCHS_MLP):
    model_mlp.train()
    idx   = np.random.permutation(N_TRAIN)
    X_tr, y_tr = X_tr[idx], y_tr[idx]
    epoch_loss = 0; n_batches = 0
    for i in range(0, N_TRAIN, BATCH):
        Xb = Tensor(X_tr[i:i+BATCH])
        yb = y_tr[i:i+BATCH]
        out  = model_mlp(Xb)
        loss = crit_mlp(out, yb)
        opt_mlp.zero_grad()
        loss.backward()
        opt_mlp.step()
        epoch_loss += loss.item(); n_batches += 1
    mlp_train_loss.append(epoch_loss / n_batches)

    model_mlp.eval()
    out_te   = model_mlp(Tensor(X_te))
    preds_te = np.argmax(out_te.data, axis=1)
    mlp_test_acc.append((preds_te == y_te).mean() * 100)

    print(f"  Epoch {epoch+1:2d}/{EPOCHS_MLP} — Loss: {mlp_train_loss[-1]:.4f} | Test Acc: {mlp_test_acc[-1]:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("MLP Classifier — Synthetic 64-dim / 10-class Benchmark", color="#c9d1d9",
             fontsize=14, fontweight="bold")

ax = axes[0]
ax.plot(range(1, EPOCHS_MLP+1), mlp_train_loss, color=ACCENT, linewidth=2, label="Train Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Training Loss", color="#c9d1d9"); ax.grid(True)
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

ax = axes[1]
ax.plot(range(1, EPOCHS_MLP+1), mlp_test_acc, color=ACCENT2, linewidth=2, label="Test Accuracy")
ax.axhline(y=10, color=ACCENT3, linewidth=1, linestyle='--', label="Random baseline (10%)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
ax.set_title("Test Accuracy", color="#c9d1d9"); ax.grid(True)
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
ax.text(0.97, 0.15, f"Peak: {max(mlp_test_acc):.1f}%",
        transform=ax.transAxes, ha='right', va='bottom', fontsize=11, color=ACCENT2,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#30363d'))

plt.tight_layout()
plt.savefig("results/mlp_benchmark.png", bbox_inches='tight')
plt.close()
print("  Saved results/mlp_benchmark.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Autograd graph visualisation (conceptual)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating autograd diagram ...")

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
ax.set_title("Computational Graph & Backpropagation", color="#c9d1d9",
             fontsize=14, fontweight="bold", pad=15)

def draw_node(ax, x, y, label, color, textcolor='white', radius=0.55):
    circle = plt.Circle((x, y), radius, color=color, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=9,
            color=textcolor, fontweight='bold', zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color, label='', fwd=True):
    dx, dy = x2-x1, y2-y1
    length = (dx**2+dy**2)**0.5
    ux, uy = dx/length, dy/length
    r = 0.58
    sx, sy = x1+ux*r, y1+uy*r
    ex, ey = x2-ux*r, y2-uy*r
    c = ACCENT if fwd else ACCENT3
    ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle='->', color=c, lw=1.8))
    if label:
        mx, my = (sx+ex)/2, (sy+ey)/2
        offset = 0.25
        ax.text(mx + offset*(-uy), my + offset*ux, label,
                ha='center', va='center', fontsize=7.5, color=c)

nodes = {
    'x':    (1.5, 5.0, '#1f6feb',   'x\n[3×4]'),
    'W':    (1.5, 2.5, '#388bfd',   'W\n[4×8]'),
    'xW':   (3.5, 3.8, '#3fb950',   'x@W\n[3×8]'),
    'b':    (3.5, 1.8, '#388bfd',   'b\n[8]'),
    'z':    (5.5, 3.0, '#3fb950',   'z = xW+b\n[3×8]'),
    'relu': (7.5, 3.0, '#d2a8ff',   'ReLU\n[3×8]'),
    'loss': (9.5, 3.0, '#f78166',   'Loss\n[ ]'),
}
for key, (x, y, c, lbl) in nodes.items():
    draw_node(ax, x, y, lbl, c)

edges_fwd = [('x','xW'), ('W','xW'), ('xW','z'), ('b','z'), ('z','relu'), ('relu','loss')]
for a, b in edges_fwd:
    x1,y1,*_ = nodes[a]; x2,y2,*_ = nodes[b]
    draw_arrow(ax, x1, y1, x2, y2, ACCENT)

edges_bwd = [('loss','relu'), ('relu','z'), ('z','x'), ('z','W'), ('z','b')]
grad_labels = ['∂L/∂relu', '∂L/∂z', '∂L/∂x', '∂L/∂W', '∂L/∂b']
for (a, b), lbl in zip(edges_bwd, grad_labels):
    x1,y1,*_ = nodes[a]; x2,y2,*_ = nodes[b]
    draw_arrow(ax, x1+0.1, y1-0.1, x2+0.1, y2-0.1, ACCENT3, label=lbl)

ax.text(6.0, 5.5, '→  Forward pass', color=ACCENT,  fontsize=11, fontweight='bold')
ax.text(6.0, 5.0, '→  Backward pass', color=ACCENT3, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("results/autograd_graph.png", bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved results/autograd_graph.png")

print("\nAll results generated successfully in results/")
