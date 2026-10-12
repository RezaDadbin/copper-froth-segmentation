# Copper Froth Segmentation — Paper Code

Python code for the planned data paper on annotated copper-froth images from
Sungun. Training, prediction, evaluation, postprocessing, motion experiments and
paper figures are organized as reusable modules and command-line scripts.

The layout follows the local **Context-Aware Neural Subdivision** repository:
explicit configs, separate implementation code, small scripts, documentation and
tests. No notebook execution order is required by the maintained pipeline.

## Layout

```text
configs/                  Eight source-derived model configurations (JSON)
froth/
  data/                   Manifest-driven TIFF/LabelMe loading and transforms
  models/                 Original U-Net, CNN builders and SAM-family adapters
  engine/                 Training, saved-mask prediction and evaluation
  postprocessing/         ROI watershed and LoG/Voronoi kernels
  motion/                 Block matching, optical flow, phase correlation, fusion
  visualization/          Model panels, comparisons, curves and motion figures
  reporting/              Tag summaries, annotation statistics and paper charts
scripts/                  Linked command-line entry points
legacy/                   All 35 notebooks + 3 original scripts exported to .py
                          (including checkpoint variants and empty sources)
tests/                    Geometry, metrics, models, pipeline and motion tests
docs/                     Reproduction, changes, support and provenance
requirements/             Shared and optional dependency groups
local_archive/            Byte-exact original files; ignored by Git
```

The active repository contains **no `.ipynb` files**. Every original code file has
an explicit mapping in [python_conversion.csv](docs/python_conversion.csv).
Historical exports preserve alternate drivers, widget visualizations, Markdown
and raw cells, while disabling IPython/package-install commands. They import
without running experiments. Use `scripts/` for supported execution.

## Install and check

Python 3.11 is the locally checked environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python tools/validate_repository.py
python -m unittest discover -s tests -v
```

Install the optional [dependency groups](docs/environment.md) for DeepLabV3+,
SAM/HQ-SAM or postprocessing/motion. Script `--help` works without ML libraries.

## Train → predict → evaluate → render

From this repository, your current project inputs are `../Dataset` and `../tags.csv`.
For another machine, replace those paths with the deposited dataset and manifest.
The [data layout](data/README.md) specifies the paired image/JSON format.

```bash
python scripts/run_pipeline.py --config configs/unet.json \
  --dataset-root ../Dataset --manifest ../tags.csv \
  --output results/generated/unet --device auto
```

The pipeline trains once, loads its best checkpoint, predicts the source `test`
images, evaluates saved binary masks, and exports model figures. Add
`--checkpoint /path/to/weights.pt` to skip training. Commands are also independent:

```bash
python scripts/train.py --config configs/unet.json \
  --dataset-root ../Dataset --manifest ../tags.csv --output results/generated/unet/training
python scripts/predict.py --config configs/unet.json \
  --checkpoint results/generated/unet/training/best.pt \
  --dataset-root ../Dataset --manifest ../tags.csv --output results/generated/unet/predictions
python scripts/evaluate.py --predictions results/generated/unet/predictions \
  --dataset-root ../Dataset --manifest ../tags.csv --output results/generated/unet/evaluation
python scripts/render_results.py --run U-Net=results/generated/unet/predictions \
  --dataset-root ../Dataset --manifest ../tags.csv --output results/generated/unet/figures
```

For comparisons, repeat `--run LABEL=PATH`. Choose exact images with
`--images 'B50 (1).tiff'`. Figures export to **PNG at 300 dpi and PDF**. Crops are
marked in native-image comparisons; probabilities and binary masks are saved
separately from display colormaps. See [all commands](docs/commands.md).

## Models

| Family | Maintained Python route | Configuration |
| --- | --- | --- |
| U-Net | Original architecture + shared trainer/inference | `configs/unet.json` |
| SBSNet | Explicit external architecture adapter; two binary output channels | `configs/sbsnet.json` |
| DeepLabV3 | torchvision ResNet50, matching supplied source | `configs/deeplabv3.json` |
| DeepLabV3+ | SMP ResNet50, matching supplied source | `configs/deeplabv3plus.json` |
| SAM | Fine-tuning modes, box inference, original AMG | `configs/sam.json` |
| HQ-SAM | Fine-tuning, intermediate-feature decoder, point-grid proposals | `configs/hq_sam.json` |
| MedSAM | Decoder training and explicitly declared box prompts | `configs/medsam.json` |
| FCN | Exploratory ResNet50 binary segmentation | `configs/fcn.json` |
| RobustSAM / SAMUS | Complete source conversion; original attempts remain incomplete | `legacy/`; `scripts/inspect_external.py` |

Base checkpoints and external implementations are not bundled. SBSNet requires
`--external-root /path/to/SBSNet-pytorch`; SAM-family models require a matching
base checkpoint. Prompted SAM inference requires `--prompt-source full_image` or
`ground_truth`. The latter is marked as annotation-assisted in saved metadata.
[Model support](docs/model_support.md) explains these requirements and the missing
separate tiled-MedSAM implementation.

## Statistics and paper figures

The existing statistical and evaluation programs are integrated as Python modules:

```bash
python scripts/dataset_statistics.py --dataset-root ../Dataset --tags ../tags.csv \
  --output-dir results/generated/statistics
python scripts/summarize_tags.py --project-root .. --output-dir results/generated/tag_analysis
python scripts/make_paper_figures.py --project-root .. --output-dir results/generated/paper_figures
```

The first command computes per-annotation and per-image geometry from JSONs with
progress monitoring. Add `--test-five` to measure exactly five real annotations.
The latter commands read the existing project CSVs; they do not train models.
Paper charts include the original Section 5/6 figures and all 14 tag-group charts.
Their input layout is documented in [commands](docs/commands.md).

## Reproduction scope

This is a **refactored implementation for new runs**, not evidence that the saved
historical CSVs were regenerated. [Changes](docs/changes.md) records loader,
threshold, protocol and postprocessing corrections. The original 237 source
files remain unchanged and are preserved byte-for-byte in the local archive.

The manifest has 595 images: 485 training and 110 **reused validation** images.
The latter were used for selection/tuning and reporting. Historical DeepLab CSV
names differ from the supplied ResNet50 code; RobustSAM identity and tiled-MedSAM
run provenance remain unresolved. Different evaluated grids also limit direct
comparison of the historical distance metrics. See [reproduction notes](docs/reproduction.md).

Author/title/DOI metadata and the project license still need their final release
values; [citation template](CITATION.cff.example) and [model references](docs/references.md)
are retained. Dataset files, model checkpoints, generated results and the local
source archive are excluded from this code repository. Dot-prefixed files and
folders are also excluded from the published tree.
