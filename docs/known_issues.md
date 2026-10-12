# Historical issues and current status

The original source remains unchanged in the archive and is exported under
`legacy/`. These findings do not establish which source generated a saved CSV.
The maintained code corrections are documented in [changes](changes.md).

| Historical issue | Maintained Python status | Remaining paper evidence |
| --- | --- | --- |
| SBSNet architecture absent inside `codes` | Explicit adapter; local external architecture found | Exact benchmark implementation/revision still needed |
| SBSNet test loader used training dataset | Shared runner selects separate source `test` samples | Older displayed validation curves remain unverified |
| Bilinear annotation-mask resize / polygon-only loaders | Shared nearest-neighbor loader supports the circle | Historical mask protocol remains a run-provenance question |
| Multiple SAM drivers and later HQ patches | Explicit fine-tuning mode and shared adapter | Exact historical execution path/checkpoint |
| DeepLab source says ResNet50, CSV names differ | New configs use ResNet50 accurately | CSV-producing backbone/stride run |
| MedSAM annotation boxes, no tiled source | Explicit prompt source; no invented tiled implementation | Full-image/tiled CSV protocol |
| RobustSAM ordinary-SAM fallback / missing medsam import | No fallback in maintained factory; original attempts preserved | Verified RobustSAM implementation and run |
| SAMUS failed setup/model construction | Python source retained, external inspection command | Completed model driver if it belongs in the paper |
| Postprocessing placeholder import / filename substitution | Linked imports and exact saved prediction filenames | Historical postprocessing-result provenance |
| LoG labels merged after binary relabeling | Region identities retained; regression tested | New count results require their own saved run |
| Unconditional postprocessing plots | Computation separated from headless rendering | None for script organization |
| Motion scaling and frame order unclear | Explicit pair/calibration inputs; synthetic shift tests | Measured spatial/time calibration for physical claims |
| Complete historical environment locks missing | Verification versions documented | Original training environment and upstream revisions |

The source `test` split is reused validation, as confirmed by the dataset owner.
All current saved historical metrics still retain their original values.
