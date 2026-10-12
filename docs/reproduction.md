# Reproduction and verification

The maintained workflow is documented in [commands](commands.md), and the code
changes in [changes](changes.md). All original source variants are Python exports
under `legacy/`, with exact mappings and hashes in `python_conversion.csv`.
Original byte-level preservation is independently checked against
`source_manifest.csv` when the original/archive folders are available.

Configs are starting specifications for new runs. Historical CSV-producing run
settings, exact external revisions and learned weights are not established by
source names. DeepLab source backbones differ from recorded CSV labels; the
RobustSAM attempts contain a fallback/import failure, and no separate tiled
MedSAM implementation was supplied. See [model support](model_support.md).

The 110 source `test` images were used for checkpoint/settings selection and
reporting, so the manuscript should describe them as reused validation. New
scripts retain source split names for file mapping and record that role. Different
crop/resolution/proposal/prompt protocols are saved in run metadata; a mean HD
in one model's grid is not physically comparable to another grid's mean HD.
Semantic foreground union metrics do not validate individual bubble instances.

## Checks

```bash
python tools/validate_repository.py
python -m unittest discover -s tests -v
```

The first command uses only the standard library: all public Python syntax,
config files, script help, relative documentation links, historical export hashes,
and the complete original/archive checksums when locally present. It can also
check a clean exported repository without the ignored archive.

The functional tests use small synthetic images. They cover polygon/circle
rasterization, paired-file mapping, binary transforms and inverse crop coverage,
analytic metrics and empty cases, U-Net forward/backward, a real small SAM decoder
when the optional package is installed, a complete train/predict/evaluate/render
cycle, annotation-change rejection, watershed/LoG region regressions, and known
motion shifts. They do not retrain the paper's full experiments.

Additional local verification reads real dataset samples, runs available CNN
forward passes without pretrained downloads, checks the existing reporting CSVs,
and regenerates paper charts into ignored verification outputs. Exact executed
checks and limitations are recorded in `verification_report.json` after validation.
A missing optional package/checkpoint remains an untested integration, regardless
of whether its wrapper compiles.
