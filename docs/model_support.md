# Model implementation and support

Maintained entry points are `scripts/train.py`, `predict.py`, `evaluate.py` and
`render_results.py`, backed by the shared `froth` package. Configurations describe
new runs; they do not recover missing historical run settings.

| Model | Source and maintained route | External requirement / limit |
| --- | --- | --- |
| U-Net | Exact original class extracted to `froth/models/unet.py` | Input/crop dimensions divisible by 32 |
| SBSNet | External class adapter in `froth/models/factory.py` | Set external root containing `net/sbsnet/SBSNet.py`; that local repo was found outside `codes` |
| DeepLabV3 | torchvision `deeplabv3_resnet50`, replaced two-class head | Source uses ResNet50; historical ResNet101 CSV mapping is unresolved |
| DeepLabV3+ | SMP `DeepLabV3Plus`, ResNet50, default output stride 16 | Source does not establish the ResNet152/OS8 CSV-producing run |
| FCN | torchvision ResNet50 with original single-channel binary head | Exploratory; no comparable recorded metric CSV |
| SAM | Official registry + gradient-preserving decoder path and AMG | Matching base model checkpoint and original SAM package |
| HQ-SAM | HQ registry, intermediate encoder features, HQ decoder, point-grid/NMS proposals | Matching HQ checkpoint/package; full HQ package integration requires those files |
| MedSAM | SAM architecture with MedSAM base weights and box-prompt training | Ground-truth prompts are explicit; separate tiled source is absent |
| RobustSAM | Both original attempts fully exported under `legacy/robustsam` | Setup has an ordinary-SAM fallback; separate inference imports a missing `medsam` API; no complete verified driver was supplied |
| SAMUS | Setup/inspection source exported under `legacy/samus` | Saved model construction failed; no completed training/inference pipeline was supplied |

For SBSNet, the local architecture found is from a coal-froth SBSNet codebase.
It is used through an explicit external path, without copying unlicensed third-party
files into this repository or claiming it produced the copper benchmark CSV.
Obtain/use the exact implementation appropriate to the reported run and record
its revision. The configured two-channel BCE target duplicates the binary mask,
as in the supplied notebook; inference averages those channel probabilities.

RobustSAM and SAMUS remain source conversions of incomplete attempts. The supported
inspection command reads their source layout/checkpoint keys; it does not substitute
SAM or fabricate missing training code. Their original code and all source cells
are accessible as Python, including saved exploratory visualization code.

Fine-tuning modes for SAM/HQ-SAM/MedSAM are `decoder_only`, `decoder+prompt`,
`prompt_only`, `encoder_only` and `full` in `training.fine_tune`. Batch size is one,
with optional gradient accumulation. Box inference distinguishes full-image
prompts from ground-truth-derived prompts in run metadata and figure labels.

Strict loading accepts raw model state dictionaries, common full-checkpoint
wrappers, and legacy `mask_decoder` checkpoints on a compatible initialized SAM
base model. It rejects incompatible model tensors instead of silently loading a
partial, differently named architecture.
