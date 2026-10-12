# Commands and connected outputs

All maintained scripts resolve implementation imports relative to their repository;
inputs and outputs are explicit arguments. Run `python scripts/NAME.py --help` for
all options. The optional installed `froth-*` commands call the same functions.

| Script | Input | Output / next consumer |
| --- | --- | --- |
| `train.py` | Model JSON, dataset, tags manifest | `best.pt`, `last.pt`, `history.csv`, `run.json`, curves |
| `predict.py` | Config + compatible checkpoint; SAM base weights when applicable | Numeric `masks/`, `scores/`, `predictions.csv`, `run.json` |
| `evaluate.py` | Saved prediction run + unchanged image/JSON pairs | Per-image metrics and split-wise mean/std/median CSVs |
| `render_results.py` | One or more saved prediction runs + dataset | Input/GT/prediction/contour/error/probability panels, native comparisons, region-label panels |
| `render_training.py` | `history.csv` | Loss, foreground IoU and pixel-accuracy curves |
| `run_pipeline.py` | Config, data; optional existing checkpoint | Connected training, prediction, evaluation and rendering directories |
| `postprocess.py` | Saved numeric score run | New prediction-compatible run + region counts; use the same eval/render scripts |
| `analyze_motion.py` | Two explicit, ordered frames | Vector CSV, motion summary, arrows and alignment/residual figures |
| `dataset_statistics.py` | LabelMe JSONs selected by tags manifest | All existing annotation/count/area/shape statistics, with progress |
| `summarize_tags.py` | Existing project tags and historical model CSVs | Per-tag and per-domain mean/std/median summaries |
| `make_paper_figures.py` | Existing project metric/statistics/analysis CSVs | 25 tables + 24 figure candidates (PNG/PDF) and captions |
| `check_foreground_baseline.py` | Historical model CSV directory | Constant-foreground references and confusion-metric checks |
| `inspect_external.py` | RobustSAM/SAMUS source directory; optional checkpoint | Source/checkpoint inspection; no guessed architecture or package installation |

## SAM-family examples

```bash
python scripts/train.py --config configs/sam.json \
  --base-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --dataset-root ../Dataset --manifest ../tags.csv --output results/generated/sam/training
python scripts/predict.py --config configs/sam.json \
  --base-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --checkpoint results/generated/sam/training/best.pt --mode amg \
  --dataset-root ../Dataset --manifest ../tags.csv --output results/generated/sam/amg
```

Replace the config and base checkpoint for HQ-SAM or MedSAM. Semantic box-prompt
inference requires `--prompt-source full_image` or `--prompt-source ground_truth`.
`--use-base-weights` explicitly requests inference without a fine-tuned checkpoint.
MedSAM does not expose an invented automatic/tiled implementation.

## Postprocessing and motion examples

```bash
python scripts/postprocess.py --predictions results/generated/unet/predictions \
  --method watershed --output results/generated/unet/watershed
python scripts/evaluate.py --predictions results/generated/unet/watershed \
  --dataset-root ../Dataset --manifest ../tags.csv --output results/generated/unet/watershed_metrics
python scripts/analyze_motion.py --first /path/to/frame1.tiff --second /path/to/frame2.tiff \
  --method fusion --output results/generated/motion
```

LoG/Voronoi needs `--reference-run` pointing to a **training-only** prediction run
made on the same model grid. Its size priors are not fitted to validation images.
Motion outputs use pixels/frame; optional physical speed requires both a measured
`--spatial-scale` in mm/px and `--seconds-per-frame`. Frame numbers alone do not
establish that two images are consecutive frames.

## Reporting input bundle

`--project-root` for the historical reporting commands is a data/report bundle:

```text
PROJECT/
  tags.csv
  tags/*.csv
  models/*.csv
  analysis/group_model_summary.csv
  tags and their accuracy/TAG/MODEL.csv
  statistically reports/
    image_statistics.csv
    froth_measurements.csv
    dataset_summary.csv
    run_report.json
```

`summarize_tags.py` needs only the first three inputs. `make_paper_figures.py` also
cross-checks the previously generated per-tag/analysis/statistical exports shown
above. They already exist in the current paper project. Place a compatible saved
report bundle at any path and point `--project-root` there. These historical CSVs
use their own original metric conventions; new `evaluate.py` CSVs have a separately
documented schema and are not silently substituted into that bundle.

Add `--csv-only` to the paper-figure command for calculations without plotting.
The dataset statistics code uses the standard library; its default data folder
is repository `data/`, and explicit input/output flags are recommended.

## Result format

A prediction directory contains `run.json` (config, versions, manifest/checkpoint
hashes, prompt source and per-image crop/resize geometry), `predictions.csv`
(image/annotation hashes and exact file mappings), `masks/IMAGE_NAME.png`
(binary 0/255) and `scores/IMAGE_NAME.npz` (numeric probability and optional
proposal/region labels). Original filename extensions are retained in output
names to prevent stem collisions. AMG scores are explicitly labeled as binary
proposal unions, rather than calibrated foreground probabilities.

Evaluation refuses changed manifests, images, annotations and incompatible grids.
Means are unweighted per-image means, separated by source split. HD/HD95 use
foreground boundaries on each saved evaluation grid; HD95 pools bidirectional
nearest-boundary distances. Empty/empty IoU and Dice are 1; undefined precision,
recall and one-empty boundary distances are blank and have defined-value counts.
