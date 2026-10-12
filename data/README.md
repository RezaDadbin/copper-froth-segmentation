# Data inputs

The dataset is not bundled with this code. Point `--dataset-root` to a directory
containing `train1/` and `test1/`, and `--manifest` to the combined `tags.csv`:

```text
DATASET/
  train1/  image.tif or image.tiff + image.json
  test1/   image.tif or image.tiff + image.json
```

`tags.csv` must include `image_name` and `split` (`train` or `test`); the paper's
full manifest also carries the domain flag and nine binary condition tags. Exact
names, including spaces, parentheses and extensions, are retained. Only listed
images are loaded; extra JSONs are excluded. Annotation `imagePath` is not used
to locate image files. Native annotation/image dimensions must agree.

The current dataset has 595 images (485 training, 110 reused validation). Current
local command paths from the repo are `--dataset-root ../Dataset --manifest ../tags.csv`.
Portable ignored storage can be `data/raw/`; no symlink to a private absolute
machine path is required. Final release URL/license will accompany the dataset.
