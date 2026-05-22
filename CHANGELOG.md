# Changelog

## [1.0.0] — 2026-05-22

### Added
- FastAPI 服务入口 `main.py`，集成 MeSquare 监控标准端点
- `POST /train` 接口：上传正常图像训练 PCA 子空间模型
- `POST /detect` 接口：上传待测图进行异常检测，返回异常分数和热力图
- `POST /reset` 接口：重置检测器状态
- `GET /status` 接口：查看当前训练状态和模型信息
- 可视化测试页面 `static/index.html`，白色极简主题
- 支持三种可视化模式：热力图叠加 (overlay)、左右对比 (side_by_side)、缺陷框标注 (bbox)
- 支持四种评分方法：重建误差 (reconstruction)、马氏距离 (mahalanobis)、欧氏距离 (euclidean)、余弦距离 (cosine)
- Dockerfile，基于 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- README.md 和 CHANGELOG.md

### Changed
- 更新 requirements.txt，添加 FastAPI 和监控依赖
- Dockerfile 重构：分层 COPY、添加构建注释、配置清华 pip 镜像源
- **移除 HuggingFace 自动下载回退**：本地模型不存在时改为抛出 `FileNotFoundError`，提示手动放置
- 移除 test.py（已由可视化测试页面替代）
