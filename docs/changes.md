# Changes from the research sources

The byte-exact originals remain in `local_archive/original_codes/`; `legacy/`
contains import-safe Python transcripts of **all 38 code files** (35 notebooks,
three Python files, including duplicates/checkpoints/empty sources). Markdown/raw
cells become comments. IPython/shell/setup lines become commented `pass`
statements, and notebook wildcard imports are delayed inside the manual historical
function. Top-level training, cloning, package uninstalling, renaming and display
calls do not execute on import. Original cell order is retained.

The maintained `froth/` modules and `scripts/` deliberately refactor those sources:

- A single manifest-based TIFF/LabelMe loader supports `.tif`/`.tiff`, polygons and
  the circle annotation, uses paired file stems rather than stale `imagePath`,
  rejects missing/duplicate pairs, and excludes unlisted annotation files.
- Masks resize with nearest-neighbor interpolation and remain binary. Historical
  U-Net/FCN adaptive center crops and the SBSNet fixed crop are explicit configs;
  native comparison panels mark pixels outside a model crop.
- U-Net architecture tensor names/shapes/forward logic are preserved. The shared
  trainer keeps batch/channel dimensions explicit, uses sigmoid probabilities
  with a configurable threshold, and fixes the SBSNet validation-loader mistake.
  Duplicate SBSNet binary channels remain explicit rather than becoming an
  invented background/foreground softmax target.
- CNN and SAM-family drivers share checkpoint/history/provenance handling.
  Training uses float32, explicit seeds, validation under no-grad, configurable
  accumulation with correct final partial-group scaling, and optional clipping.
  Original mixed-precision branches are retained in the historical exports.
  DeepLab class weights are estimated deterministically from the first up-to-200
  listed training samples, and its polynomial schedule updates per epoch.
- SAM alternatives become one selected fine-tuning mode. SAM-family image
  normalization precedes padding through the model's preprocess method; HQ
  intermediate encoder features are passed explicitly. These standardized new
  runs do not recover undocumented historical cell execution orders.
- MedSAM's source BCE loss is retained. Its normalization/padding now uses the
  shared SAM path; evaluation prompts must be named explicitly. No tiled
  inference implementation is invented from a CSV filename.
- Binary masks, probability maps and display colormaps are separate artifacts.
  Evaluation and rendering read the exact same prediction index/geometry.
  New metric definitions and empty-mask handling are documented in commands.md;
  historical metric CSVs are not overwritten or reinterpreted as new outputs.
- ROI watershed computation is separated from unconditional plotting. The
  historical mean-minus-k-times-standard-deviation threshold is bounded to the
  valid probability range; uniform maps use 0.5. This prevents a negative
  threshold from converting zero-valued background into foreground.
- LoG/Voronoi filtering relabels surviving label IDs instead of merging adjacent
  regions back into one binary connected component. Scale priors use an explicit
  training-only prediction run. Region counts remain exploratory.
- Motion functions accept explicit frames and paths. Signed-SAD fixes unsigned
  subtraction, NCC follows the fusion notebook, and sparse smoothing divides by
  its support instead of attenuating displacement with inserted zeros.
  Directional coherence uses a circular resultant in [0,1], rather than the
  notebook's inverse linear angle variance. Physical calibration is explicit.
- Existing annotation statistics, tag-analysis calculations and Section 5/6
  figure definitions are integrated with explicit paths. Paper-figure boxplot
  keyword selection supports both locally available Matplotlib APIs. The
  constant-foreground checker is now a callable Python module/CLI.

These changes make code paths reviewable for **new runs**. They do not establish
that repaired code, discovered external architectures or revised thresholds
produced the previously supplied benchmark CSVs. Original datasets, model CSVs,
figures and source files were not modified during conversion.
