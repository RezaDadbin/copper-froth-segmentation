#!/usr/bin/env bash
# Inference and post-processing on the evaluation images, in the order used for the paper. Run from the repository root.
# Models whose fine-tuned checkpoint is not in $FROTH_CKPT can be dropped from the lists; the zero-shot models need only
# the public weights downloaded by setup_gpu.sh.
set -euo pipefail
SPLIT="${1:-test1}"

# semantic routes: froth probability maps
python -m paper.inference.infer --split "$SPLIT" \
  --models dlv3,dlv3p,sam_ft,hq_ft,medsam_ft,medsam_ft_fix,sam_zs,robust_zs
# automatic routes: instance label maps
python -m paper.inference.infer --split "$SPLIT" --amg \
  --models amg_sam_zs,amg_sam_ft,amg_sam_ft_raw,amg_hq_ft,amg_robust_zs,amg_sam_h_zs,amg_hq_h_zs,amg_sam2_zs,cellpose_zs
# ROI watershed and LoG-seeded Voronoi on every semantic output (CPU)
python -m paper.inference.postprocess
