"""
================================================================
CHART 5 — Confusion Matrix (Entity-Level Heatmap)
Shows which entity types get confused with each other
HireSense AI | BERT + BiLSTM + CRF
================================================================
Install: pip install matplotlib numpy seaborn
Run    : python chart5_confusion_matrix.py
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── REPLACE with your actual confusion matrix counts ──────────
# Rows = True labels, Cols = Predicted labels
# Order must match 'labels' list below

labels = ["SKILL","EXP","EDU","PROJ","ACH","CERT","ORG","LOC","DATE","NAME","CONTACT","SECTOR"]

# Simulated realistic confusion matrix (diagonal = correct predictions)
cm = np.array([
    [430,  4,  0,  6,  2,  3,  5,  0,  0,  0,  0,  2],  # SKILL
    [  3,310,  2,  8,  5,  1, 12,  2,  0,  0,  0,  4],  # EXP
    [  0,  2,280,  1,  0,  3,  4,  0,  0,  0,  0,  0],  # EDU
    [  5,  9,  1,220,  6,  0,  4,  0,  0,  0,  0,  1],  # PROJ
    [  2,  4,  0,  7,160,  2,  2,  0,  0,  0,  0,  0],  # ACH
    [  3,  1,  2,  0,  2,190,  1,  0,  0,  0,  0,  0],  # CERT
    [  4, 10,  3,  3,  1,  1,240,  5,  0,  0,  0,  3],  # ORG
    [  0,  2,  0,  0,  0,  0,  6,180,  2,  0,  0,  0],  # LOC
    [  0,  0,  0,  0,  0,  0,  0,  2,260,  0,  0,  0],  # DATE
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,210,  4,  0],  # NAME
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,200,  0],  # CONTACT
    [  1,  3,  0,  1,  0,  0,  2,  0,  0,  0,  0,130],  # SECTOR
])
# ─────────────────────────────────────────────────────────────

# Normalize row-wise (recall per class)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5,
    linecolor="#1a1b2e",
    ax=ax,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 9, "color": "black"}
)

ax.set_title("Entity Confusion Matrix (Row-Normalized)\nHireSense AI — BERT+BiLSTM+CRF",
             fontsize=14, color="white", fontweight="bold", pad=15)
ax.set_xlabel("Predicted Label", fontsize=12, color="white", labelpad=10)
ax.set_ylabel("True Label", fontsize=12, color="white", labelpad=10)
ax.tick_params(colors="white", labelsize=9)

# Style colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors="white")
cbar.ax.yaxis.label.set_color("white")

plt.tight_layout()
plt.savefig("chart5_confusion_matrix.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → chart5_confusion_matrix.png")
plt.show()
