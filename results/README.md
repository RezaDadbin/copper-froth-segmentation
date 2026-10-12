# Generated results

All new results belong under ignored `results/generated/` (or an explicit external
output path). The linked pipeline creates `training/`, `predictions/`, `evaluation/`
and `figures/`. See [commands](../docs/commands.md) for their schemas.

Original figures and notebook outputs remain in the byte-exact local archive;
conversion does not replace them or regenerate the historical benchmark. The
parent project's existing CSVs remain the inputs to paper-reporting commands.
Verification artifacts are labeled separately under `results/generated/verification/`.
