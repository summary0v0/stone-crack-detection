# 验证样图

仅保留两张用于检查程序运行的图片，不保留完整训练集或标注集。

| 本机文件 | 原相对路径 | SHA-256 |
| --- | --- | --- |
| `marble_sample.jpg` | `dataset/images/_1024_1536_20210531_10560.jpg` | 见 `manifest.json` |
| `marble_crack.jpg` | `dataset/images/_768_2304_20210531_10554.jpg` | 见 `manifest.json` |

来源：MachineLearningVisionRG 的 [Marble Crack Segmentation (MCS) 数据集](https://github.com/MachineLearningVisionRG/mcs-dataset)，本项目原有副本于 2026-10-17 下载。样图未修改像素，授权为 CC BY 4.0，完整许可保留在本目录 `LICENSE`。

原数据集说明引用：Vrochidou et al. (2022), *Towards Robotic Marble Resin Application: Crack Detection on Marble Using Deep Learning*, Electronics 11(20), 3289, DOI: [10.3390/electronics11203289](https://doi.org/10.3390/electronics11203289)。原始石材图像数据由 Aman Rastogi 发布于 [Marble Surface Anomaly Detection - 2](https://www.kaggle.com/datasets/wardaddy24/marble-surface-anomaly-detection-2)。

样图用于证明加载、推理、显示及保存链路可运行，不代表识别精度验收。
