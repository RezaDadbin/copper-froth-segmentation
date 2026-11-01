# Copper froth segmentation

Code for segmenting bubbles in flotation-froth images: training and inference for convolutional and SAM-family
models, classical post-processing from froth masks to individual bubbles, froth motion analysis, dark-impurity
detection, and the code behind the results of our paper (in preparation):

> R. Dadbin, S. Lotfi, A. Sokhandan. *Deep segmentation of industrial copper flotation froth: a review of data,
> models and processing pipelines, with a comparative study on Sungun plant images.*

The froth images used in the paper are the property of the Sungun copper company and are not distributed. The code
works with any froth image set annotated in [LabelMe](https://github.com/wkentaro/labelme) with the layout described
in [docs/data_format.md](docs/data_format.md).

## Contents

| Path | Content |
| --- | --- |
| `froth/` | Python package: data loading, models, training, prediction, evaluation, post-processing, motion, impurities, annotation statistics |
| `scripts/` | Command-line entry points for the package |
| `configs/` | Model and pipeline configurations |
| `paper/` | Code behind every result, figure and table of the paper ([paper/README.md](paper/README.md)) |
| `tests/` | Unit tests on synthetic data |
| `docs/` | [Commands](docs/commands.md) and [data format](docs/data_format.md) |

## Installation

Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

Optional dependency groups are in `requirements/`: `cnn.txt` (DeepLabV3+), `sam.txt` (SAM family),
`postprocessing_motion.txt` (post-processing, motion and impurity detection), `paper.txt` and `paper_gpu.txt`
(reproduction of the paper).

## Usage

Train, predict, evaluate and render one model:

```bash
python scripts/run_pipeline.py --config configs/unet.json \
  --dataset-root ../Dataset --manifest ../tags.csv --output outputs/unet --device auto
```

The steps can also be run separately with `scripts/train.py`, `scripts/predict.py`, `scripts/evaluate.py` and
`scripts/render_results.py`; add `--checkpoint path/to/weights.pt` to skip training. Further tools:

```bash
python scripts/postprocess.py --help            # ROI watershed and LoG-seeded Voronoi: from probability maps to bubbles
python scripts/analyze_motion.py --help         # froth motion between two frames
python scripts/register_frames.py f1.png f2.png # multiscale phase-correlation registration with residuals
python scripts/froth_stability.py f1.png f2.png # fused local motion and tile-wise stability
python scripts/detect_impurities.py --folder path/to/images --output outputs/impurities
python scripts/dataset_statistics.py --help     # bubble size and shape statistics from the annotations
```

See [docs/commands.md](docs/commands.md) for all commands and options.

## Models

| Model | Implementation | Configuration |
| --- | --- | --- |
| U-Net | `froth/models/unet.py` | `configs/unet.json` |
| SBSNet | Adapter for the authors' implementation (not redistributed; set `--external-root`) | `configs/sbsnet.json` |
| DeepLabV3, DeepLabV3+ | torchvision, segmentation-models-pytorch | `configs/deeplabv3.json`, `configs/deeplabv3plus.json` |
| SAM, HQ-SAM, MedSAM | Official registries; decoder fine-tuning, box prompts, automatic mask generation | `configs/sam.json`, `configs/hq_sam.json`, `configs/medsam.json` |
| FCN | torchvision | `configs/fcn.json` |

The exact model variants of the paper (DeepLabV3 ResNet-101, DeepLabV3+ ResNet-152 at output stride 8, and the
zero-shot SAM ViT-H, HQ-SAM ViT-H, SAM 2.1, RobustSAM and Cellpose-SAM) are run by `paper/inference`.
SAM-family models need their public base weights, passed with `--base-checkpoint`; no weights are included.

## Reproducing the paper

```bash
bash paper/inference/setup_gpu.sh && bash paper/inference/run_inference.sh   # GPU
python -m paper.evaluation.metrics && python -m paper.evaluation.analysis      # CPU
```

[paper/README.md](paper/README.md) lists every step and maps each figure and table to the command that produces it.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

[MIT](LICENSE)
