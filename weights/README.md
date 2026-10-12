# Checkpoints

Model weights are external and ignored by Git. The source `codes` folder did not
include the model weight files referenced by its experiments.

Set `model.base_checkpoint` in SAM-family JSON configs, or pass
`--base-checkpoint /path/to/weights.pth`. Relative checkpoint paths in the provided
configs resolve from the repository root. Training saves full `best.pt` and
`last.pt` state dictionaries and `history.csv` beside a run record.

Prediction requires `--checkpoint` for trained weights, or explicitly
`--use-base-weights` for a SAM-family base-model run. Decoder-only legacy SAM
checkpoints require a compatible base model. Loading is strict; record checkpoint
hashes and the implementation revision with any claimed paper result.
