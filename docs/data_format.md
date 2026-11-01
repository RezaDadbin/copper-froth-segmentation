# Data format

The images of the paper are not distributed. The code works with any froth image set in the following layout.
Pass the image folder with `--dataset-root` and the manifest with `--manifest`:

```text
DATASET/
  train1/  image.tif (or .tiff) + image.json
  test1/   image.tif (or .tiff) + image.json
tags.csv
```

- Annotations are LabelMe JSON files with the same name as the image; bubbles are polygons labelled `froth`.
- `tags.csv` needs the columns `image_name` and `split` (`train` or `test`); further columns, such as condition
  tags, are optional.
- Only images listed in the manifest are loaded. Images and annotations are paired by file name, not by the
  `imagePath` field, and their sizes must agree.
