# 必要官方网络源码

- 仓库：<https://github.com/LouisNUST/CrackFormer-II>
- 提交：`1d701aea7b09e85b106a74894cf4779f59fe9af6`
- 权重来源和 SHA-256：上一级 `provenance.json`。

本项目仅保留原网络及其导入依赖，文件内容未改动：

| 文件 | SHA-256 |
| --- | --- |
| `CrackFormer-II/nets/crackformerII.py` | `d7fe74a7b04794c3309bd3abdf86749517c7799c0c269c7843e87df8a7d81ad6` |
| `CrackFormer-II/utils/lossFunctions.py` | `48e9c616254d45f999b8482730fbfb689d410e9a11e51ce2b3ff3ad6b31c1909` |

网络文件在模块导入时引用 `utils.lossFunctions`，所以后者仍需保留。完整训练集、数据加载器、其它网络、训练日志和原验证入口均不属于本机推理运行链。
