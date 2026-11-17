# Multimodal Tea Quality Prediction

English | [简体中文](README_zh.md)

This release provides **processed data from tea gardens 2, 3, and 4** used in our multimodal tea quality prediction research, together with data loading, integrity checking, and group-splitting utilities. Contact the author for additional regions and other research data.

## Public data

| Garden | Identifier | Samples |
| --- | --- | ---: |
| 2 | `2_MH_guangjingmeng` | 3 |
| 3 | `3_MH_mangda` | 5 |
| 4 | `4_MH_8haoshuiku` | 5 |
| Total | 3 tea gardens | 13 |

Each sample contains processed inputs and quality labels used in the research:

- **Multispectral:** cropped and resized `MSS.npy`, shape `(4, 45, 110)`, in **Green, NIR, Red, RedEdge** band order.
- **Leaf RGB:** processed `front_rgb_image.npy` and `back_rgb_image.npy`, each `(3, 224, 224)`, with ImageNet channel normalization already applied.
- **Numerical inputs:** altitude (1), soil measurements (16), weather indices (6), and chlorophyll/nitrogen summary statistics (12).
- **Quality labels:** 30 targets; their order and names are in [`schema.json`](data/processed/schema.json).

The release contains **39 `.npy` files**. Multispectral arrays retain their processed snapshot's original numerical scale; tabular inputs and quality labels also retain their original scales. Raw aerial imagery, original leaf photographs, source weather spreadsheets, intermediate experimental files, and other gardens are excluded.

## Layout

```text
data/processed/
├── samples.csv                 Sample IDs, numerical inputs, labels, array paths
├── schema.json                 Column order, names, shapes, processing notes
├── manifest.json               Release scope and file SHA-256 checksums
├── 2_MH_guangjingmeng/
│   ├── Q01/
│   │   ├── MSS.npy
│   │   ├── front_rgb_image.npy
│   │   └── back_rgb_image.npy
│   └── ...
├── 3_MH_mangda/
└── 4_MH_8haoshuiku/
scripts/
├── dataset.py                  Loading, verification, and grouped splits
├── inspect_data.py             Dataset summary and integrity checks
├── make_splits.py              Demonstration garden-held-out folds
└── export_processed.py         Author-side export from a local processed snapshot
```

`sample_id` preserves the source identifier; `region` identifies the tea garden. Array paths are relative to `data/processed/`. The CSV uses UTF-8 with BOM; its row order is preserved by the loader. Numerical columns are `soil_01..16`, `weather_01..06`, `lcc_n_01..12`, `altitude`, and `quality_01..30`.

## Quick start

Use Python 3.9 or later:

```bash
git clone https://github.com/GuoquanPei/Multimodal-Tea-Quality-Prediction.git
cd Multimodal-Tea-Quality-Prediction
python -m pip install -r requirements.txt
python scripts/inspect_data.py
python scripts/make_splits.py --output outputs/garden_splits.json
python -m unittest discover -s tests -v
```

Read one multispectral array:

```python
import numpy as np

mss = np.load("data/processed/2_MH_guangjingmeng/Q01/MSS.npy", allow_pickle=False)
print(mss.shape)  # (4, 45, 110), channels first
```

Read all public samples and labels:

```python
from scripts.dataset import load_dataset

samples = load_dataset()  # SHA-256 verification included; 13 samples
sample = samples[0]
print(sample["sample_id"], sample["MSS"].shape, sample["quality"].shape)
```

Public data loading requires neither PyTorch nor pickle nor the full local research workspace. RGB arrays are already normalized: do not divide by 255 or apply ImageNet normalization again. For display, recover RGB in `[0, 1]` with `normalized * std + mean`, broadcasting `mean=[0.485,0.456,0.406]` and `std=[0.229,0.224,0.225]` over channels.


## Data and code availability

The processed data for these three gardens and the accompanying data scripts are publicly available in this repository. **For additional regions, further research materials, or other data requests, please contact the author.** Include the intended research question and requested data scope; the author will explain what can be provided and how to obtain it. This notice does not guarantee the scope or timing of access to unreleased materials.

**Affiliation:** College of Big Data, Yunnan Agricultural University

**Contact:** Guoquan Pei

**Email:** [peiguoquan@outlook.com](mailto:peiguoquan@outlook.com)

## Citation and license

Please acknowledge this repository and the commit used when reusing the public data or scripts. The complete paper citation will be added after publication. The repository retains its existing [Apache-2.0 license](LICENSE).
