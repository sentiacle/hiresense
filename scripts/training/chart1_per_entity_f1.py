"""
================================================================
CHART 1 — Per-Entity F1 Score Bar Chart
(Precision, Recall, F1 for each of 12 NER entity types)
HireSense AI | BERT + BiLSTM + CRF
================================================================
Install: pip install matplotlib numpy
Run    : python chart1_per_entity_f1.py
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── REPLACE these with your actual seqeval classification_report numbers ──
entities   = ["SKILL", "EXP", "EDU", "PROJ", "ACH", "CERT",
              "ORG",   "LOC", "DATE","NAME","CONTACT","SECTOR"]

precision  = [0.91, 0.87, 0.93, 0.84, 0.79, 0.88, 0.82, 0.90, 0.94, 0.89, 0.92, 0.76]
recall     = [0.89, 0.83, 0.91, 0.80, 0.74, 0.85, 0.78, 0.87, 0.92, 0.86, 0.90, 0.71]
f1_scores  = [0.90, 0.85, 0.92, 0.82, 0.76, 0.86, 0.80, 0.88, 0.93, 0.87, 0.91, 0.73]
# ─────────────────────────────────────────────────────────────────────────

x      = np.arange(len(entities))
width  = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

bars_p = ax.bar(x - width, precision, width, label="Precision", color="#4C9BE8", alpha=0.9)
bars_r = ax.bar(x,          recall,   width, label="Recall",    color="#56C596", alpha=0.9)
bars_f = ax.bar(x + width,  f1_scores,width, label="F1 Score",  color="#F4845F", alpha=0.9)

# Value labels on top of each bar
for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=6.5, color="white", fontweight="bold")

ax.set_xlabel("Entity Type", fontsize=12, color="white", labelpad=10)
ax.set_ylabel("Score", fontsize=12, color="white", labelpad=10)
ax.set_title("Per-Entity Precision, Recall & F1 Score\nHireSense AI — BERT+BiLSTM+CRF",
             fontsize=14, color="white", fontweight="bold", pad=15)

ax.set_xticks(x)
ax.set_xticklabels(entities, color="white", fontsize=9)
ax.set_ylim(0.0, 1.08)
ax.tick_params(colors="white")
ax.spines[:].set_color("#333344")
ax.yaxis.grid(True, color="#1e2030", linestyle="--", linewidth=0.7)
ax.set_axisbelow(True)

legend = ax.legend(fontsize=10, facecolor="#1a1b2e", edgecolor="#333344",
                   labelcolor="white", loc="lower right")

plt.tight_layout()
plt.savefig("chart1_per_entity_f1.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → chart1_per_entity_f1.png")
plt.show()
