#!/usr/bin/env bash
# Install the GPU inference dependencies and download the public weights (SAM, HQ-SAM, SAM 2, RobustSAM, Cellpose-SAM).
# Run from the repository root. The fine-tuned checkpoints of the study are not distributed (see paper/README.md).
set -euo pipefail
WORK="${FROTH_WORK:-outputs}"
mkdir -p "$WORK" "${FROTH_CKPT:-$WORK/ckpt}"
CKPT="$(cd "${FROTH_CKPT:-$WORK/ckpt}" && pwd)"
ROBUSTSAM="${FROTH_ROBUSTSAM:-$(cd "$WORK" && pwd)/RobustSAM}"

pip install -q -r requirements/paper_gpu.txt
SAM2_BUILD_CUDA=0 pip install -q --no-build-isolation git+https://github.com/facebookresearch/sam2.git \
  || SAM2_BUILD_CUDA=0 pip install -q git+https://github.com/facebookresearch/sam2.git

cd "$CKPT"
for url in https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
           https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
           https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
           https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth \
           https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth; do
  wget -q -nc "$url"
done
python -c "from huggingface_hub import hf_hub_download as d; d('robustsam/robustsam', 'model_checkpoint/robustsam_checkpoint_b.pth', local_dir='$CKPT')"
[ -d "$ROBUSTSAM" ] || git clone -q https://github.com/robustsam/RobustSAM "$ROBUSTSAM"
python -c "from cellpose import models; models.CellposeModel(gpu=True)" > /dev/null 2>&1   # caches the generalist model
echo "Weights in $CKPT"
