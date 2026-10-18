# 本机石材裂纹检测

- 本项目仅运行本机 Stone331 / CrackFormer-II 推理；不依赖远程服务器，不读取 SSH 配置，不启动训练。
- 使用根目录 `.venv/Scripts/python.exe`。根目录 `.python` 是此虚拟环境的基础运行时，不能作为缓存删除。
- 保持 CUDA 版 PyTorch；设备支持 `auto`、`cpu`、`cuda`。自动模式根据本机 CUDA 可用性选择；显式 GPU 模式不可用时应报错。
- `crackformer_ii/` 存放界面、推理程序、必要网络源码、权重和来源记录；`.agent/examples/` 只保留少量带来源说明的验证样图。
- `crackformer_ii/outputs/` 存放本机检测结果；`.agent/reports/` 保留当前环境和功能验证记录。Agent 验证材料统一放在 `.agent/`，不在根目录重新创建 `examples/` 或 `reports/`。
- 保留权重 SHA-256、严格加载、输入及输出校验；不得用部分加载掩盖模型不匹配。
- 本机 CPU 验证不代表 RTX 4060 实机 CUDA 验证或识别精度达标；当前能力为石材裂纹分割。
- 清理、移动和生成文件限制在本项目目录内，保留用户输入原图及其他项目的环境。
