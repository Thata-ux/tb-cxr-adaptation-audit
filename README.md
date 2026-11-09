# Higher AUC, Wrong Reasons? Lung-Masked Auditing of Tuberculosis Domain Adaptation

Code and results for a study of test-time and few-shot domain adaptation for tuberculosis (TB)
detection on chest X-rays, transferring between the public **Shenzhen (SZ)** and **Montgomery (MC)**
datasets. Besides the usual AUC, every adapted model is audited with a **Lung Attention Ratio (LAR)**
computed from Grad-CAM and lung masks, and with a **lung-occlusion test**.

**Main findings**

- Few-shot fine-tuning with 20 labelled target images per class gave the most consistent AUC
  (0.841–0.891, vs. 0.689–0.821 without adaptation).
- Changes in AUC were uncorrelated with changes in LAR (Spearman ρ = −0.08, p = 0.14, n = 360);
  24% of configurations that improved AUC *reduced* attention to the lungs.
- With the lungs blurred, models still reached a mean AUC of 0.645 — they rely on cues outside the lungs.
- TENT collapses to AUC ≈ 0.52 when evaluated on label-sorted batches (see *Note on TENT* below).

## Repository layout

```
notebooks/
  01_main_run_v2.ipynb      Full experiment (80 source models + oracle), Kaggle notebook
  02_tent_rerun_v3.ipynb    Rerun of source-only + TENT with shuffled evaluation batches
scripts/
  merge_results.py          Reproducibility check, merge of both runs, tables and figures
  make_figures.py           Figures 1–2 of the manuscript from results/deltas.csv
results/                    Final (merged) results used in the manuscript
  metrics.csv               One row per backbone × direction × arm × seed × method
  deltas.csv                ΔAUC / ΔLAR of every method vs. source-only
  table_summary.csv         Mean ± SD of all metrics
  table_tests.csv           DeLong (per seed) and Wilcoxon (across seeds) tests
  table_occlusion.csv       Occlusion test summary
  wrong_reasons_rate.csv    Share of "higher AUC, lower LAR" configurations
  raw/predictions.csv       Per-image predicted probabilities
  raw/lar_per_image.csv     Per-image LAR and enrichment
figures/
  manuscript/               Figures 1–3 of the paper
  supplementary/            Label-efficiency curves and occlusion bar charts
```

## Data

The images are **not** included. Both datasets are public and were released by the U.S. National
Library of Medicine. The notebooks read the Kaggle collection
[Chest Xray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels),
which contains the SZ and MC images together with lung masks. Files are matched by their names
(`CHNCXR_xxxx_y`, `MCUCXR_xxxx_y`; the last digit is the label, 1 = TB).

| Dataset | Images | Normal | TB | Test images (after a 20-per-class few-shot pool) |
|---|---|---|---|---|
| Shenzhen (SZ) | 662 | 326 | 336 | 622 (530 with lung masks) |
| Montgomery (MC) | 138 | 80 | 58 | 98 (all with lung masks) |

## Reproducing the results

The experiments were run on Kaggle (one NVIDIA T4 GPU).

1. Create a Kaggle notebook, import `notebooks/02_tent_rerun_v3.ipynb`, attach the dataset above,
   and enable **GPU** and **Internet** (ImageNet weights are downloaded through `timm`).
2. In the configuration cell set `RERUN_TENT_ONLY = False` and `QUICK = False`, then
   *Save Version → Save & Run All*. This runs the complete, corrected experiment in one pass
   (about 6–7 hours) and writes everything to `/kaggle/working/results/`.
   Set `QUICK = True` first for a ~15-minute smoke test.

The published results were produced in two steps, which are kept for transparency:
`01_main_run_v2.ipynb` (all methods), then `02_tent_rerun_v3.ipynb` with `RERUN_TENT_ONLY = True`
(source-only + TENT only). The source-only AUCs of both runs are identical in all 80
configurations, so the TENT rows were replaced with:

```bash
pip install -r requirements.txt
python scripts/merge_results.py --main path/to/results --rerun path/to/results_tent_rerun --out results
python scripts/make_figures.py --results results --out figures/manuscript
```

## Note on TENT

TENT normalises features with the statistics of the current batch at prediction time. In the main
run the target images were ordered by label, so each batch held a single class and the class signal
was normalised away (AUC ≈ 0.52 everywhere). Version 3 shuffles the evaluation order with a fixed
seed and restores the original order afterwards. Anyone evaluating batch-statistics methods should
make sure batches are not sorted by class.

## Experimental grid

- Backbones: DenseNet-121, EfficientNet-B0 (ImageNet-pretrained, `timm`)
- Directions: SZ→MC, MC→SZ
- Source-training strategies: baseline, CLAHE, histogram matching, MixStyle
- Adaptation: source-only, AdaBN, TENT, SHOT, few-shot fine-tuning (k = 5, 10, 20), oracle (5-fold CV)
- Seeds: 0–4 → 80 source models + 4 oracle configurations

## Citation

If you use this code or these results, please cite the paper (details will be added after publication)
and the original datasets:

- S. Jaeger *et al.*, "Two public chest X-ray datasets for computer-aided screening of pulmonary
  diseases," *Quant. Imaging Med. Surg.*, vol. 4, no. 6, pp. 475–477, 2014.
- S. Jaeger *et al.*, "Automatic tuberculosis screening using chest radiographs," *IEEE Trans. Med.
  Imaging*, vol. 33, no. 2, pp. 233–245, 2014.
- S. Candemir *et al.*, "Lung segmentation in chest radiographs using anatomical atlases with
  nonrigid registration," *IEEE Trans. Med. Imaging*, vol. 33, no. 2, pp. 577–590, 2014.

## License

Code: MIT (see `LICENSE`). Results and figures: CC BY 4.0.
