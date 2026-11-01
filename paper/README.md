# Reproducing the paper

This folder contains the code behind every result, figure and table of our paper (in preparation):

> R. Dadbin, S. Lotfi, A. Sokhandan. *Deep segmentation of industrial copper flotation froth: a review of data,
> models and processing pipelines, with a comparative study on Sungun plant images.*

The froth images and annotations are the property of the Sungun copper company and are not distributed, and neither
are the checkpoints fine-tuned on them. The code runs on any froth image set with the same layout (below); the
zero-shot models need only public weights, so their part of the study can be repeated on other data.

## Data layout and locations

All scripts are run from the repository root as modules (`python -m paper....`). Locations are read from environment
variables; the defaults assume the private folders sit next to the repository.

| Variable | Default | Content |
| --- | --- | --- |
| `FROTH_DATA` | `../Dataset` | `train1/`, `test1/`: images (`.tif`, `.tiff` or `.png`) with LabelMe JSON files of the same name (polygons labelled `froth`) |
| `FROTH_TAGS` | `../tags.csv` | `split, image_name, domain_yellow_copper` and the nine binary condition tags |
| `FROTH_RECORDED` | `../models` | per-image metric CSVs saved by the original training runs (`metrics_*.csv`) |
| `FROTH_SAVED_CNN` | `../saved_cnn` | saved U-Net and SBSNet output maps (`model_output_U_Net/`, `model_ouptut_SBS_Net/`) |
| `FROTH_WORK` | `outputs` | inference outputs (`out/`), metrics and analysis results |
| `FROTH_CKPT` | `outputs/ckpt` | model weights |
| `FROTH_FIGURES`, `FROTH_TABLES` | `outputs/figures`, `outputs/tables` | generated figures (PDF/PNG) and LaTeX table bodies |

## 1. Inference (GPU)

```bash
bash paper/inference/setup_gpu.sh       # dependencies and public weights (SAM, HQ-SAM, SAM 2, RobustSAM, Cellpose-SAM)
bash paper/inference/run_inference.sh   # all models on test1, then both post-processing routes
```

Inference only: no model is trained. Semantic routes write a froth probability map per image (uint8 PNG) and automatic
routes an instance label map (uint16 PNG) to `$FROTH_WORK/out/<model>/<split>/`. Every fine-tuned model receives
exactly the preprocessing it was trained with, including the details documented in Section 5.4 of the paper.

| Key | Model | Weights in `$FROTH_CKPT` |
| --- | --- | --- |
| `dlv3`, `dlv3p` | DeepLabV3 ResNet-101; DeepLabV3+ ResNet-152, output stride 8 | `dlv3_r101_ep9.pth`, `dlv3p_r152_ep48.pth` (fine-tuned, not distributed) |
| `sam_ft`, `amg_sam_ft`, `amg_sam_ft_raw` | SAM ViT-B, fine-tuned decoder (box / automatic / automatic with training normalisation) | `sam_vit_b_decoder_only_decoder_best.pth` (not distributed) |
| `hq_ft`, `amg_hq_ft` | HQ-SAM ViT-B, fine-tuned decoder | `sam_hq_vit_b_decoder_best.pth` (not distributed) |
| `medsam_ft`, `medsam_ft_fix` | MedSAM, fine-tuned decoder (training-time / corrected up-sampling) | `medsam_vit_b_decoder_only_full_best.pth` (not distributed) |
| `sam_zs`, `amg_sam_zs`, `amg_sam_h_zs` | SAM ViT-B (box, automatic), SAM ViT-H (automatic), zero-shot | public |
| `amg_hq_h_zs` | HQ-SAM ViT-H, automatic, zero-shot | public |
| `amg_sam2_zs` | SAM 2.1 Hiera-L, automatic, zero-shot | public |
| `robust_zs`, `amg_robust_zs` | RobustSAM ViT-B (box, single-prompt point grid), zero-shot | public |
| `cellpose_zs` | Cellpose-SAM generalist model, zero-shot | public |

## 2. Metrics and analyses (CPU)

```bash
python -m paper.evaluation.metrics       # per-image pixel, boundary, instance, count and size metrics -> metrics.csv
python -m paper.evaluation.analysis      # bootstrap CIs, domains, conditions, recorded-run reconciliation
python -m paper.evaluation.reliability   # bubble size, matching threshold, counts, size distributions
python -m paper.evaluation.saved_cnn     # saved U-Net / SBSNet maps on the same crops as the other models
```

All bootstrap intervals use 2,000 resamples of the images with a fixed seed; rows are sorted before resampling, so
the intervals are reproducible.

## 3. Figures and tables

```bash
python -m paper.figures.review                 # Figure 2
python -m paper.figures.annotation             # Figure 3
python -m paper.figures.pipeline_thumbnails    # image thumbnails of Figure 4
python -m paper.figures.preprocessing          # Figure 5 and Figure S14
python -m paper.figures.outputs main post saved   # Figures 6, 7 and S12-S13
python -m paper.figures.supplementary          # Figures S1-S11
python -m paper.figures.data_tables --measurements <froth_measurements.csv>   # Tables 7, 8, 11
python -m paper.figures.review_tables          # Tables S6, S7
```

`froth_measurements.csv` is written by `python scripts/dataset_statistics.py --dataset-root ../Dataset --tags ../tags.csv --output-dir outputs/statistics`. Figures 1 and 4 are drawn in LaTeX; the
counts of Figure 1 are in `review/search.json`.

| Paper item | Produced by | Output |
| --- | --- | --- |
| Figure 2 | `paper.figures.review` | `review_overview.pdf` |
| Figure 3 | `paper.figures.annotation` | `annotation.pdf` |
| Figure 5, Figure S14 | `paper.figures.preprocessing` | `preprocessing.pdf`, `recorded_runs.pdf` |
| Figures 6, 7 | `paper.figures.outputs main post` | `model_outputs.pdf`, `postprocessing.pdf` |
| Figure 8 | `paper.evaluation.analysis` | `condition_instance.pdf` |
| Figures 9, 10 | `paper.evaluation.reliability` | `size_outcomes.pdf`, `reliability.pdf` |
| Figures S1-S11, S12-S13 | `paper.figures.supplementary`, `paper.figures.outputs saved` | `supplementary/S*.png` |
| Tables 7, 8, 11 | `paper.figures.data_tables` | `tag_counts.tex`, `geometry.tex`, `recorded_runs.tex` |
| Table 12 | `paper.evaluation.saved_cnn` | `saved_cnn.tex` |
| Table 13, Tables S1-S3 | `paper.evaluation.analysis` | `eval110.tex`, `eval110_ci_a.tex`, `eval110_ci_b.tex`, `eval110_colour.tex`, `eval110_grey.tex` |
| Table 14, Tables S4-S5 | `paper.evaluation.reliability` | `count_abs.tex`, `size_by_domain.tex`, `iou_sens.tex` |
| Tables S6, S7 | `paper.figures.review_tables` | `all_studies.tex`, `not_tabulated.tex` |

## Literature review records

`review/records.json` holds the 68 reviewed records with the fields used in the paper (data, method, test protocol,
instance-level metrics and the level of access to the source); `review/search.json` holds the search queries, the
screening criteria, the PRISMA counts and the eligible studies that could not be tabulated.
