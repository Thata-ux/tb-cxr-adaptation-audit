#!/usr/bin/env python3
"""Recreate Figures 1 and 2 of the manuscript from results/deltas.csv.

Figure 3 (Grad-CAM examples) is produced by section 13 of the notebooks because it
needs a trained model and the images.

Usage: python scripts/make_figures.py --results results --out figures/manuscript
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="figures/manuscript")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---- Figure 1: research pipeline
fig, ax = plt.subplots(figsize=(7.2, 2.9)); ax.set_xlim(-1.5, 101.5); ax.set_ylim(4, 40); ax.axis("off")
def box(x, y, w, h, title, lines, fc="#f2f2f2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#333", lw=0.9))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=8.2, weight="bold")
    ax.text(x + w / 2, y + h - 6.8, "\n".join(lines), ha="center", va="top", fontsize=7, linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1))
box(0, 6, 17, 32, "Public data", ["Shenzhen (SZ)", "662 CXRs", "", "Montgomery (MC)", "138 CXRs", "", "Lung masks", "", "SZ→MC, MC→SZ"])
box(21, 6, 19, 32, "Source training", ["DenseNet-121", "EfficientNet-B0", "", "Baseline", "CLAHE", "Histogram matching", "MixStyle", "", "5 seeds"])
box(44, 6, 19, 32, "Adaptation", ["Source-only", "AdaBN", "TENT", "SHOT", "Few-shot", "k = 5, 10, 20", "", "Oracle (5-fold CV)"])
box(67, 23, 33, 15, "Performance", ["AUC, Sens@Spec70,", "Spec@Sens90, ECE", "DeLong, Wilcoxon"], fc="#ffffff")
box(67, 6, 33, 15, "Attention audit", ["Grad-CAM → LAR, enrichment", "Occlusion: blur outside lungs", "/ blur lungs; Spearman ΔAUC–ΔLAR"], fc="#ffffff")
arrow(17.8, 22, 20.6, 22); arrow(40.8, 22, 43.6, 22); arrow(63.8, 26, 66.6, 30); arrow(63.8, 18, 66.6, 14)
fig.tight_layout(pad=0.1); fig.savefig(os.path.join(args.out, "fig1_pipeline.png"), dpi=300); plt.close(fig)

# ---- Figure 2: change in AUC vs change in LAR
d = pd.read_csv(os.path.join(args.results, "deltas.csv")).dropna(subset=["dAUC", "dLAR"])
names = {"adabn": "AdaBN", "tent": "TENT", "shot": "SHOT", "fewshot_k5": "Few-shot k = 5",
         "fewshot_k10": "Few-shot k = 10", "fewshot_k20": "Few-shot k = 20"}
from scipy import stats
rho, p = stats.spearmanr(d.dAUC, d.dLAR)
up = d[d.dAUC > 0.02]; share = (up.dLAR < -0.02).mean()
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.fill_between([-0.12, -0.02], 0.02, 0.36, color="#d62728", alpha=0.07, lw=0)
for (k, lab), mk in zip(names.items(), ["o", "s", "^", "D", "v", "P"]):
    s = d[d.method == k]; ax.scatter(s.dLAR, s.dAUC, s=16, alpha=.7, marker=mk, label=lab)
ax.axhline(0, c="grey", lw=.8); ax.axvline(0, c="grey", lw=.8)
ax.set_xlim(-0.12, 0.155); ax.set_ylim(-0.23, 0.36)
ax.text(-0.118, 0.352, f"Higher AUC, lower lung attention ({share:.0%})", fontsize=8, color="#a01c1c", va="top")
ax.set_xlabel("ΔLAR vs. source-only"); ax.set_ylabel("ΔAUC vs. source-only")
ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.text(1.02, 0.02, f"Spearman ρ = {rho:.2f}\np = {p:.2f}; n = {len(d)}", transform=ax.transAxes, fontsize=8, va="bottom")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig2_dauc_vs_dlar.png"), dpi=300); plt.close(fig)
print("Figures written to", args.out)
