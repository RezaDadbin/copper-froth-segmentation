# Commands

Run any script with `--help` for all options. After `pip install -e .`, the `froth-*` commands call the same
functions.

| Script | Input | Output |
| --- | --- | --- |
| `train.py` | Model config, dataset, manifest | `best.pt`, `last.pt`, `history.csv`, `run.json`, training curves |
| `predict.py` | Config and checkpoint (SAM-family: also the base weights) | `masks/`, `scores/`, `predictions.csv`, `run.json` |
| `evaluate.py` | Prediction run, dataset, manifest | Per-image metrics and per-split mean, standard deviation and median |
| `render_results.py` | One or more prediction runs, dataset | Image, ground truth, prediction, contour, error and probability panels |
| `render_training.py` | `history.csv` | Loss, IoU and pixel-accuracy curves |
| `run_pipeline.py` | Config, dataset, manifest; optional checkpoint | Training, prediction, evaluation and figures in one run |
| `postprocess.py` | Prediction run | Instance labels (ROI watershed or LoG-seeded Voronoi), in the same format as a prediction run |
| `analyze_motion.py` | Two ordered frames | Displacement vectors, motion summary and figures |
| `register_frames.py` | Two frames | Multiscale phase-correlation shift with residual diagnostics |
| `froth_stability.py` | Two frames | Fused local motion and tile-wise stability maps |
| `detect_impurities.py` | Image or folder | Impurity masks, overlays and a table of counts per image |
| `dataset_statistics.py` | LabelMe annotations and manifest | Bubble counts, sizes and shape statistics |

## SAM-family models

Download the public base weights of SAM, HQ-SAM or MedSAM and pass them with `--base-checkpoint`:

```bash
python scripts/train.py --config configs/sam.json --base-checkpoint path/to/sam_vit_b_01ec64.pth \
  --dataset-root ../Dataset --manifest ../tags.csv --output outputs/sam/training
python scripts/predict.py --config configs/sam.json --base-checkpoint path/to/sam_vit_b_01ec64.pth \
  --checkpoint outputs/sam/training/best.pt --mode amg \
  --dataset-root ../Dataset --manifest ../tags.csv --output outputs/sam/amg
```

Use `configs/hq_sam.json` or `configs/medsam.json` with their base weights for the other models. Box-prompt
inference needs `--prompt-source full_image` or `--prompt-source ground_truth`; `--use-base-weights` runs the base
model without a fine-tuned checkpoint. MedSAM supports box prompts only.

## Post-processing and motion

```bash
python scripts/postprocess.py --predictions outputs/unet/predictions \
  --method watershed --output outputs/unet/watershed
python scripts/evaluate.py --predictions outputs/unet/watershed \
  --dataset-root ../Dataset --manifest ../tags.csv --output outputs/unet/watershed_metrics
python scripts/analyze_motion.py --first frame1.tiff --second frame2.tiff \
  --method fusion --output outputs/motion
```

LoG-seeded Voronoi needs `--reference-run`, a prediction run over training images on the same grid, from which
the bubble-size priors are estimated. Motion is reported in pixels per frame; give `--spatial-scale` (mm/px) and
`--seconds-per-frame` for physical speeds.

## Annotation statistics

```bash
python scripts/dataset_statistics.py --dataset-root ../Dataset --tags ../tags.csv --output-dir outputs/statistics
```

## Output format

A prediction run contains `run.json` (configuration, package versions, manifest and checkpoint hashes, prompt
source and per-image crop geometry), `predictions.csv` (image and annotation hashes and file mapping),
`masks/IMAGE_NAME.png` (binary 0/255) and `scores/IMAGE_NAME.npz` (probabilities and, where applicable, proposal
or region labels). For automatic mask generation, the score is the union of the binary proposals, not a calibrated
probability.

Evaluation checks that the manifest, images and annotations are unchanged since prediction. Means are unweighted
per-image means within each split. HD and HD95 are computed on the foreground boundaries of the evaluation grid;
HD95 pools the nearest-boundary distances in both directions. When both masks are empty, IoU and Dice are 1;
undefined precision, recall and boundary distances are left blank.
