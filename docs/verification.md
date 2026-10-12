# Conversion verification

The [machine-readable report](verification_report.json) records executed checks
and remaining optional integrations. Source-only checks are in
[repository_validation.json](repository_validation.json).

- All 35 notebooks and three Python sources have import-safe `.py` exports; the
  clean public repository contains no notebooks.
- All 237 original/archive files match their recorded hashes. The extracted
  U-Net class has the same executable AST as its original notebook class.
- All 20 functional tests passed, including an actual small training → prediction
  → saved-mask evaluation → PNG/PDF rendering cycle and real small-SAM gradients.
- Five real dataset images loaded with binary masks across the configured grids;
  DeepLabV3, FCN and external SBSNet passed finite CPU forward checks without
  pretrained downloads. The statistics CLI passed its five-annotation check.
- Tag summary CSVs match the prior export byte-for-byte. All 25 paper tables agree;
  24 are byte-identical and the remaining size-summary differences are at most
  7.28e-12 absolute (1.12e-15 relative), consistent with Python-version summation.
  All 24 figure candidates regenerated as 48 PNG/PDF files; 155 input hashes
  remained unchanged. A representative chart was visually inspected.
- A clean Git-eligible export passes validation with no archive, dataset or
  weights. These conversion checks were completed before repository publication.

HQ-SAM/SMP full integrations and full-size pretrained/checkpoint runs were not
executed because their optional packages/checkpoints were unavailable. RobustSAM
and SAMUS source attempts remain incomplete, and separate tiled-MedSAM source
was absent. These limits are documented in [model support](model_support.md).

Verification results live under ignored `results/generated/verification/`; they
are quality checks for this conversion, not newly reported paper experiments.
