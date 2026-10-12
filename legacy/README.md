# Historical Python source exports

All 35 original notebooks and all three `.py` files are represented as `.py`,
including duplicate checkpoint variants and empty sources. The mapping and hashes
are in [python_conversion.csv](../docs/python_conversion.csv). All original cell
text/order is retained as code or comments; outputs are in the local byte-exact
archive. The earlier organized notebooks also remain locally archived.

Every export defines `historical_experiment()` and has no experiment side effects
on import. Running an export directly prints a pointer to the maintained commands;
it deliberately does not launch multiple alternative training branches or mutate
packages/files. Shell/IPython setup lines are disabled. Manual historical execution
requires reviewing the original path variables, missing dependencies and cell order.

The original widget renderers and all alternative plots are preserved in these
Python transcripts. The supported rendering replacement is
`froth/visualization/` via `scripts/render_results.py`, `render_training.py`,
`analyze_motion.py` and `make_paper_figures.py`.
