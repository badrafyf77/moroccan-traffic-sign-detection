# Data preparation

These scripts turn the raw source datasets into the cleaned dataset used by the training notebook.
Run them **in order**, on a copy of your data: they edit and delete files in place.

Every label file uses the Mapillary JSON schema:

```json
{
  "width": 1920, "height": 1080, "ispano": false,
  "objects": [
    {"key": "…", "label": "regulatory--stop--g1",
     "bbox": {"xmin": 1056.0, "ymin": 676.0, "xmax": 1160.0, "ymax": 728.0},
     "properties": {"occluded": false, "out-of-frame": false, "ambiguous": false, "…": false}}
  ]
}
```

| Step | Script | What it does |
|:---:|---|---|
| 1 | `01_merge_sources.py` | Copies every `images/` and `labels/` folder found under the root into one `images/` + `labels/` pair, skipping duplicate file names |
| 2 | `02_normalize_labels.py` | Removes the sign-type prefix (`regulatory--stop--g1` → `stop--g1`) and drops boxes smaller than 50 px |
| 3 | `03_keep_target_classes.py` | Keeps only the sign classes used on Moroccan roads |
| 4 | `04_strip_version_suffix.py` | Removes the version suffix (`stop--g1` → `stop`) so design variants share one class |
| 5 | `05_remove_other_sign.py` | Drops the catch-all `other-sign` class |
| 6 | `06_remove_empty_samples.py` | Deletes images whose label file no longer contains any sign |
| – | `count_labels.py` | Writes the number of boxes per class to `label_counts.txt` |

```bash
python data_preparation/01_merge_sources.py      path/to/sources
python data_preparation/02_normalize_labels.py   path/to/sources
python data_preparation/03_keep_target_classes.py path/to/sources
python data_preparation/04_strip_version_suffix.py path/to/sources
python data_preparation/05_remove_other_sign.py  path/to/sources
python data_preparation/06_remove_empty_samples.py path/to/sources
python data_preparation/count_labels.py          path/to/sources
```

The result (`path/to/sources/images` and `path/to/sources/labels`) is the input of step 1 of
[`notebooks/train_yolo11n.ipynb`](../notebooks/train_yolo11n.ipynb).
