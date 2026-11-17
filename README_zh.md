# 多模态茶叶品质预测

[English](README.md) | 简体中文

本次公开研究中使用的 **2、3、4 三个茶园的处理后数据**，以及数据读取、完整性校验和分组划分脚本。其余地区及更多研究数据可联系作者获取。

## 公开范围

| 茶园编号 | 数据标识 | 样本数 |
| --- | --- | ---: |
| 2 | `2_MH_guangjingmeng` | 3 |
| 3 | `3_MH_mangda` | 5 |
| 4 | `4_MH_8haoshuiku` | 5 |
| 合计 | 3 个茶园 | 13 |

每个样本包含研究使用的处理后输入与品质标签：

- 多光谱：已裁剪、统一尺寸的 `MSS.npy`，形状为 `(4, 45, 110)`，波段顺序为 **Green、NIR、Red、RedEdge**。
- 叶片 RGB：`front_rgb_image.npy` 和 `back_rgb_image.npy`，形状均为 `(3, 224, 224)`，已按 ImageNet 均值和标准差归一化。
- 数值特征：海拔 1 项、土壤 16 项、气象 6 项、叶绿素/氮素统计量 12 项。
- 品质标签：30 项，顺序及名称见 [`schema.json`](data/processed/schema.json)。

共发布 **39 个 `.npy` 数组文件**。多光谱保留处理后快照的数值原尺度；环境特征和品质标签也保留原尺度。公开包不含原始航拍文件、原始叶片照片、原始气象表、实验过程文件或其他茶园的数据。

## 文件组织

```text
data/processed/
├── samples.csv                 样本、数值特征、标签和数组相对路径
├── schema.json                 字段顺序、名称、形状与处理说明
├── manifest.json               公开范围、样本数和文件 SHA-256
├── 2_MH_guangjingmeng/
│   ├── Q01/
│   │   ├── MSS.npy
│   │   ├── front_rgb_image.npy
│   │   └── back_rgb_image.npy
│   └── ...
├── 3_MH_mangda/
└── 4_MH_8haoshuiku/
scripts/
├── dataset.py                  读取、校验与分组接口
├── inspect_data.py             查看样本与数组信息
├── make_splits.py              生成按茶园留出的示例划分
└── export_processed.py         作者从本地处理后快照导出指定子集
```

`samples.csv` 的 `sample_id` 保留原快照的样本标识，`region` 为茶园标识；数组路径相对于 `data/processed/`。CSV 编码为 UTF-8（带 BOM），行顺序与读取脚本返回顺序一致。数值列为 `soil_01..16`、`weather_01..06`、`lcc_n_01..12`、`altitude`、`quality_01..30`。

## 快速使用

使用 Python 3.9 或更新版本：

```bash
git clone https://github.com/GuoquanPei/Multimodal-Tea-Quality-Prediction.git
cd Multimodal-Tea-Quality-Prediction
python -m pip install -r requirements.txt
python scripts/inspect_data.py
python scripts/make_splits.py --output outputs/garden_splits.json
python -m unittest discover -s tests -v
```

读取单个多光谱文件：

```python
import numpy as np

mss = np.load("data/processed/2_MH_guangjingmeng/Q01/MSS.npy", allow_pickle=False)
print(mss.shape)  # (4, 45, 110)，通道在前
```

读取全部公开样本和标签：

```python
from scripts.dataset import load_dataset

samples = load_dataset()  # 自动校验 SHA-256；13 个样本
sample = samples[0]
print(sample["sample_id"], sample["MSS"].shape, sample["quality"].shape)
```

公开数据读取不需要 PyTorch、pickle 或完整的本地研究项目。RGB 数组已归一化，请勿再次除以 255 或重复应用 ImageNet 归一化。显示时可用 `RGB = normalized * std + mean` 恢复到 `[0, 1]`，其中 `mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`，按通道广播。


## 数据与代码可用性

上述三个茶园的处理后数据和配套数据脚本已在本仓库公开。**更多地区的数据、其他研究材料和进一步的数据获取需求，请联系作者**。来信请说明研究问题和需要的数据范围，作者将告知可提供的内容及获取方式。本说明不承诺尚未公开材料的提供时间或范围。

**单位：** 云南农业大学 大数据学院

**联系人：** Guoquan Pei

**邮箱：** [peiguoquan@outlook.com](mailto:peiguoquan@outlook.com)

## 引用与许可

使用公开数据或脚本时，请注明本仓库及所使用的提交版本；论文正式发表后将在此补充完整引用信息。本仓库保留现有 [Apache-2.0 许可证](LICENSE)。
