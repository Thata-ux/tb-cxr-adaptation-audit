#!/usr/bin/env python3
"""Merge the main run with the TENT rerun and recompute all tables and figures.

Background: in the main run (notebooks/01_main_run_v2.ipynb) TENT was evaluated on
label-sorted batches, which collapses its AUC to ~0.52 because TENT normalises with
per-batch statistics. notebooks/02_tent_rerun_v3.ipynb fixes this (shuffled batches)
and reruns source-only + TENT for all 80 configurations.

This script
  1. checks that source-only AUCs of both runs are identical (reproducibility check),
  2. replaces the TENT rows of the main run with those of the rerun,
  3. recomputes the summary tables, statistical tests and figures.

Usage:
  python scripts/merge_results.py --main <results_dir> --rerun <results_tent_rerun_dir> --out results
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("--main", required=True)
ap.add_argument("--rerun", required=True)
ap.add_argument("--out", default="results")
args = ap.parse_args()
OUT_DIR = args.out
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. reproducibility check
k = ["backbone", "direction", "arm", "seed"]
a = pd.read_csv(os.path.join(args.main, "metrics.csv"))
b = pd.read_csv(os.path.join(args.rerun, "metrics.csv"))
j = (a[a.method == "source_only"].set_index(k).auc.rename("main")
     .to_frame().join(b[b.method == "source_only"].set_index(k).auc.rename("rerun"), how="inner"))
diff = (j["main"] - j["rerun"]).abs().max()
print(f"Source-only check: n={len(j)}, max |AUC difference| = {diff:.6f}")
if diff > 1e-6:
    sys.exit("Source-only AUCs differ between runs; refusing to merge.")

# ---- 2. merge
for f in ["metrics.csv", "predictions.csv", "lar_per_image.csv"]:
    x = pd.read_csv(os.path.join(args.main, f)); y = pd.read_csv(os.path.join(args.rerun, f))
    pd.concat([x[x.method != "tent"], y[y.method == "tent"]], ignore_index=True) \
      .to_csv(os.path.join(OUT_DIR, f), index=False)
    print(f"merged {f}")
pd.DataFrame().to_csv(os.path.join(OUT_DIR, "errors.csv"), index=False)

# ---- 3. analysis (identical to sections 11-12 of the notebooks)
# ===================== 11. ANALISIS & TABEL =====================
def delong_roc_variance(ground_truth, predictions):
    """Fast DeLong (Sun & Xu, 2014) untuk satu atau beberapa prediktor berpasangan."""
    order = (-ground_truth).argsort(); label_1_count = int(ground_truth.sum())
    preds = predictions[:, order]
    m, n = label_1_count, preds.shape[1] - label_1_count
    pos, neg = preds[:, :m], preds[:, m:]
    def midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]: j += 1
            T[i:j] = 0.5 * (i + j - 1); i = j
        T2 = np.empty(N); T2[J] = T + 1; return T2
    k = preds.shape[0]
    tx = np.array([midrank(r) for r in pos]); ty = np.array([midrank(r) for r in neg])
    tz = np.array([midrank(r) for r in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01)); sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n

def delong_test(y, p1, p2):
    aucs, cov = delong_roc_variance(np.asarray(y), np.vstack([p1, p2]))
    l = np.array([[1, -1]]); z = np.abs(np.diff(aucs)) / np.sqrt(max((l @ cov @ l.T).item(), 1e-12))
    return float(aucs[1] - aucs[0]), float(2 * stats.norm.sf(z).item())

met = pd.read_csv(os.path.join(OUT_DIR, "metrics.csv"))
if "nan_preds" in met.columns:
    print("Baris gagal (prediksi NaN) dikeluarkan dari analisis:", int(met.nan_preds.notna().sum()))
    met = met[met.nan_preds.isna()].copy()
pred = pd.read_csv(os.path.join(OUT_DIR, "predictions.csv"))

# Tabel 1: ringkasan per arah × backbone × lengan × metode (mean ± sd lintas seed)
cols = ["auc", "sens_at_spec70", "spec_at_sens90", "ece", "lar_mean", "enrich_mean",
        "auc_outside_blurred", "auc_lung_occluded"]
cols = [c for c in cols if c in met.columns]
g = met.groupby(["direction", "backbone", "arm", "method"])[cols]
summary = g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)
summary.to_csv(os.path.join(OUT_DIR, "table_summary.csv"))
try:
    display(summary)
except NameError:
    print(summary.to_string())

# Tabel 2: ΔAUC vs ΔLAR terhadap source-only (konfigurasi berpasangan)
base = met[met.method == "source_only"].set_index(["direction", "backbone", "arm", "seed"])
d = met[~met.method.isin(["source_only", "oracle_cv"])].copy()
d = d.join(base[["auc", "lar_mean", "auc_lung_occluded"]].add_suffix("_src"),
           on=["direction", "backbone", "arm", "seed"])
d["dAUC"] = d.auc - d.auc_src; d["dLAR"] = d.lar_mean - d.lar_mean_src
d["dLungOcclAUC"] = d.auc_lung_occluded - d.auc_lung_occluded_src
d.to_csv(os.path.join(OUT_DIR, "deltas.csv"), index=False)
ok = d[["dAUC", "dLAR"]].dropna()
if len(ok) > 3:
    rho, pv = stats.spearmanr(ok.dAUC, ok.dLAR)
    print(f"\nSpearman ΔAUC vs ΔLAR: rho={rho:.3f}, p={pv:.4f} (n={len(ok)})")
    # 'Higher AUC, wrong reasons' = AUC naik tetapi LAR turun
    wr = ((d.dAUC > 0.02) & (d.dLAR < -0.02)).groupby([d.direction, d.method]).mean().round(3)
    print("\nProporsi konfigurasi 'AUC naik, LAR turun':\n", wr)
    wr.to_csv(os.path.join(OUT_DIR, "wrong_reasons_rate.csv"))

# Tabel 3: Wilcoxon antar-seed (metode vs source-only) + DeLong per seed
rows = []
for (dr, bb, arm, mth), sub in d.groupby(["direction", "backbone", "arm", "method"]):
    w = stats.wilcoxon(sub.auc, sub.auc_src).pvalue if len(sub) >= 5 and (sub.auc != sub.auc_src).any() else np.nan
    dl = []
    for sd in sub.seed:
        a = pred[(pred.direction == dr) & (pred.backbone == bb) & (pred.arm == arm) & (pred.seed == sd)]
        p0 = a[a.method == "source_only"].sort_values("key"); p1 = a[a.method == mth].sort_values("key")
        if len(p0) == len(p1) and len(p0) > 0:
            dl.append(delong_test(p0.label.values, p0.prob.values, p1.prob.values)[1])
    rows.append(dict(direction=dr, backbone=bb, arm=arm, method=mth, n_seeds=len(sub),
                     dAUC_mean=sub.dAUC.mean(), wilcoxon_p=w,
                     delong_p_median=np.median(dl) if dl else np.nan,
                     delong_sig_frac=np.mean(np.array(dl) < 0.05) if dl else np.nan))
tests = pd.DataFrame(rows).round(4)
tests.to_csv(os.path.join(OUT_DIR, "table_tests.csv"), index=False)
print(tests.to_string())

# %%
# ===================== 12. GAMBAR =====================
import matplotlib.pyplot as plt
FIG = os.path.join(OUT_DIR, "figures"); os.makedirs(FIG, exist_ok=True)

# Gambar 1: ΔAUC vs ΔLAR
if len(ok) > 3:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for mth, sub in d.dropna(subset=["dAUC", "dLAR"]).groupby("method"):
        ax.scatter(sub.dLAR, sub.dAUC, label=mth, s=22, alpha=0.75)
    ax.axhline(0, c="grey", lw=0.8); ax.axvline(0, c="grey", lw=0.8)
    ax.set_xlabel("Δ Lung Attention Ratio vs source-only"); ax.set_ylabel("Δ AUC vs source-only")
    ax.set_title("Higher AUC, wrong reasons? (kuadran kiri atas)")
    ax.legend(fontsize=8, frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dauc_vs_dlar.png"), dpi=300); plt.close('all')

# Gambar 2: kurva efisiensi label (few-shot)
fs = met[met.method.str.startswith("fewshot") | (met.method == "source_only")].copy()
fs = fs[fs.arm.isin(["baseline", "mixstyle"])]
fs["k"] = fs.method.str.extract(r"k(\d+)")[0].astype(float).fillna(0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (dr, sub) in zip(axes, fs.groupby("direction")):
    for (bb, arm), s2 in sub.groupby(["backbone", "arm"]):
        agg = s2.groupby("k").auc.agg(["mean", "std"])
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], marker="o", capsize=3, label=f"{bb}/{arm}")
    orc = met[(met.method == "oracle_cv") & (met.direction == f"oracle->{dr.split('->')[1]}")]
    if len(orc): ax.axhline(orc.auc.mean(), ls="--", c="k", lw=1, label="oracle CV")
    ax.set_title(dr); ax.set_xlabel("citra berlabel target per kelas (k)"); ax.set_xticks([0, 5, 10, 20])
axes[0].set_ylabel("AUC target"); axes[-1].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_label_efficiency.png"), dpi=300); plt.close('all')

# Gambar 3: uji oklusi (AUC asli vs luar-paru-diblur vs paru-diblur)
occ = met[met.method != "oracle_cv"].groupby(["direction", "method"])[["auc_masked_subset", "auc_outside_blurred", "auc_lung_occluded"]].mean()
occ.to_csv(os.path.join(OUT_DIR, "table_occlusion.csv"))
for dr, sub in occ.groupby(level=0):
    sub = sub.droplevel(0)
    ax = sub.plot.bar(figsize=(8, 3.8), rot=30); ax.axhline(0.5, ls=":", c="grey")
    ax.set_title(f"Uji oklusi — {dr}"); ax.set_ylabel("AUC"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"fig_occlusion_{dr.replace('->', '_to_')}.png"), dpi=300); plt.close('all')

# %%
