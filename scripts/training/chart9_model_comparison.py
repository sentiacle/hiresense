"""
================================================================
CHART 9 — Model Comparison Bar Chart
Compares: Rule-Based vs BiLSTM+CRF vs BERT+CRF vs BERT+BiLSTM+CRF
Metrics  : Precision, Recall, F1
HireSense AI
================================================================
Install: pip install matplotlib numpy
Run    : python chart9_model_comparison.py
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ── REPLACE with your actual baseline comparison numbers ──────
models = [
    "Rule-Based",
    "BiLSTM+CRF\n(no BERT)",
    "BERT+CRF\n(no BiLSTM)",
    "BERT+BiLSTM+CRF\n(Ours)"
]

precision = [0.61, 0.74, 0.83, 0.89]
recall    = [0.55, 0.70, 0.80, 0.87]
f1        = [0.58, 0.72, 0.81, 0.88]
# ─────────────────────────────────────────────────────────────

x     = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# Colors: muted for baselines, vivid for "Ours"
prec_colors = ["#3a5a7c", "#3a5a7c", "#3a5a7c", "#4C9BE8"]
rec_colors  = ["#2e6e50", "#2e6e50", "#2e6e50", "#56C596"]
f1_colors   = ["#7a4030", "#7a4030", "#7a4030", "#F4845F"]

bars_p = ax.bar(x - width, precision, width, label="Precision", color=prec_colors, alpha=0.92, edgecolor="#0f1117")
bars_r = ax.bar(x,          recall,   width, label="Recall",    color=rec_colors,  alpha=0.92, edgecolor="#0f1117")
bars_f = ax.bar(x + width,  f1,       width, label="F1 Score",  color=f1_colors,   alpha=0.92, edgecolor="#0f1117")

# Value labels
for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.006,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="white", fontweight="bold")

# Highlight "Ours" column with a background rect
ax.axvspan(2.62, 3.62, color="#F4845F", alpha=0.05, zorder=0)
ax.text(3, 0.01, "★ Ours", ha="center", va="bottom",
        fontsize=8, color="#F4845F", fontweight="bold")

ax.set_xlabel("Model", fontsize=12, color="white", labelpad=10)
ax.set_ylabel("Score", fontsize=12, color="white", labelpad=10)
ax.set_title("Model Comparison — Precision, Recall & F1\nHireSense AI NER Benchmarks",
             fontsize=14, color="white", fontweight="bold", pad=15)

ax.set_xticks(x)
ax.set_xticklabels(models, color="white", fontsize=9)
ax.set_ylim(0.0, 1.05)
ax.tick_params(colors="white")
ax.spines[:].set_color("#333344")
ax.yaxis.grid(True, color="#1e2030", linestyle="--", linewidth=0.7)
ax.set_axisbelow(True)

# Custom legend (single color per metric)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4C9BE8", label="Precision"),
    Patch(facecolor="#56C596", label="Recall"),
    Patch(facecolor="#F4845F", label="F1 Score"),
]
ax.legend(handles=legend_elements, fontsize=10,
          facecolor="#1a1b2e", edgecolor="#333344",
          labelcolor="white", loc="lower right")

plt.tight_layout()
plt.savefig("chart9_model_comparison.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → chart9_model_comparison.png")
plt.show()
