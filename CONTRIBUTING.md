# Maintaining the paper code

Keep reusable implementation in `froth/`, argument parsing in `froth/cli.py`,
small command wrappers in `scripts/`, and explicit model settings in `configs/`.
Render from saved numeric results rather than requiring notebook state. Keep
native image names, prompts, crop/grid geometry and checkpoint provenance with
new outputs. Place large data/checkpoints/results outside tracked source files.

Run `python tools/validate_repository.py` and the applicable functional tests
before a change. Record scientific protocol changes in `docs/changes.md`. A new
implementation does not establish the identity of older CSV-producing runs.

`legacy/` is a historical transcript export, not the maintained implementation.
Use its source mapping to inspect alternatives. Change active modules rather
than inserting new experimental code into those exports. Keep the byte-exact
original snapshot in the ignored local archive while preparing the paper.

Final citation and license metadata remain to be supplied for publication.
