"""
================================================================
CHART 2 — Training & Validation Loss + F1 Curve
Shows loss and F1 over epochs for both train and validation
HireSense AI | BERT + BiLSTM + CRF
================================================================
Install: pip install matplotlib numpy
Run    : python chart2_loss_f1_curves.py
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── REPLACE with your actual per-epoch logs ───────────────────
epochs = [1, 2, 3, 4, 5, 6]

train_loss = [1.842, 1.204, 0.876, 0.641, 0.501, 0.423]
val_loss   = [1.631, 1.089, 0.821, 0.698, 0.653, 0.641]

train_f1   = [0.51,  0.67,  0.76,  0.82,  0.86,  0.88]
val_f1     = [0.55,  0.69,  0.78,  0.83,  0.87,  0.88]

best_epoch = 6   # epoch where best val F1 was saved
# ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 5.5))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

for ax in [ax1, ax2]:
    ax.set_facecolor("#0f1117")
    ax.spines[:].set_color("#333344")
    ax.tick_params(colors="white")
    ax.yaxis.grid(True, color="#1e2030", linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Epoch", fontsize=11, color="white", labelpad=8)

# ── Loss curve ──
ax1.plot(epochs, train_loss, "o-", color="#4C9BE8", linewidth=2.2,
         markersize=6, label="Train Loss")
ax1.plot(epochs, val_loss,   "s--",color="#F4845F", linewidth=2.2,
         markersize=6, label="Val Loss")
ax1.axvline(best_epoch, color="#56C596", linestyle=":", linewidth=1.5, alpha=0.7)
ax1.text(best_epoch + 0.05, max(train_loss)*0.95, "Best\nModel",
         color="#56C596", fontsize=8, va="top")
ax1.set_ylabel("CRF Loss", fontsize=11, color="white", labelpad=8)
ax1.set_title("Training vs Validation Loss", fontsize=12,
              color="white", fontweight="bold", pad=12)
ax1.legend(fontsize=9, facecolor="#1a1b2e", edgecolor="#333344", labelcolor="white")

# Shade gap between curves to show overfitting risk
ax1.fill_between(epochs, train_loss, val_loss,
                 alpha=0.08, color="#F4845F")

# ── F1 curve ──
ax2.plot(epochs, train_f1, "o-", color="#4C9BE8", linewidth=2.2,
         markersize=6, label="Train F1")
ax2.plot(epochs, val_f1,   "s--",color="#F4845F", linewidth=2.2,
         markersize=6, label="Val F1")
ax2.axvline(best_epoch, color="#56C596", linestyle=":", linewidth=1.5, alpha=0.7)
ax2.text(best_epoch + 0.05, min(train_f1)*1.02, "Best\nModel",
         color="#56C596", fontsize=8, va="bottom")
ax2.set_ylabel("Macro F1 Score", fontsize=11, color="white", labelpad=8)
ax2.set_title("Training vs Validation F1 Score", fontsize=12,
              color="white", fontweight="bold", pad=12)
ax2.legend(fontsize=9, facecolor="#1a1b2e", edgecolor="#333344", labelcolor="white")
ax2.set_ylim(0.40, 1.0)

# Annotate final val F1
ax2.annotate(f"Val F1 = {val_f1[-1]:.2f}",
             xy=(epochs[-1], val_f1[-1]),
             xytext=(epochs[-1]-1.8, val_f1[-1]-0.07),
             arrowprops=dict(arrowstyle="->", color="#56C596", lw=1.4),
             color="#56C596", fontsize=9)

fig.suptitle("HireSense AI — Training Dynamics (BERT+BiLSTM+CRF)",
             fontsize=14, color="white", fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig("chart2_loss_f1_curves.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → chart2_loss_f1_curves.png")
plt.show()
