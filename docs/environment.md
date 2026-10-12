# Environment

Use Python 3.11 in an isolated environment. The local functional checks used
Python 3.11.7, PyTorch 2.6.0, torchvision 0.21.0, NumPy 1.26.4, OpenCV 4.11.0,
Matplotlib 3.8.0, SciPy 1.16.2 and scikit-image 0.22.0. These describe verification,
not recovered historical training environments.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

| Dependency file | Use |
| --- | --- |
| `requirements.txt` / `requirements/base.txt` | U-Net, torchvision CNNs, evaluation and rendering |
| `requirements/cnn.txt` | Adds segmentation-models-pytorch and timm for DeepLabV3+ |
| `requirements/sam.txt` | Adds official SAM and HQ-SAM imports used by source code |
| `requirements/postprocessing_motion.txt` | Adds scikit-image and motion/postprocessing dependencies |
| `requirements/notebooks.txt` | Optional inspection of archived original notebooks; not needed by Python workflows |

The SAM Git URL follows the supplied source; it is not a recovered historical
commit. External implementations/checkpoints are documented in
[model support](model_support.md). No guessed `medsam`, `SBSNet` or placeholder
`your_main_script` PyPI dependency is added. Full training environments and external
commits should be recorded with any new paper run.

The locally available official SAM package was used for a small real decoder
gradient test. HQ-SAM and SMP were not installed in the local verification
interpreter; their complete integration therefore remains dependent on the
optional environment and checkpoints. No system packages were changed.
