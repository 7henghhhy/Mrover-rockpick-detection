# Colab 标注及训练交接

## 这份 notebook 做什么

这是训练前的数据标注流程：CoTracker3 根据一次点击跟踪五个关键点，生成 YOLO Pose 格式的标签。它没有 YOLO 训练、验证或模型部署代码。请从根目录 README 的 Colab 按钮打开 notebook。

## 准备与逐视频操作

1. 在 Google Drive 建立项目目录，视频放进 `Videos/`。将 Cell 1 的 `PROJECT_DIR` 改成实际目录，输出 ZIP 会写到同级 `Annotated/`。
2. 在 Colab 选择 T4 GPU，运行 Cell 1 和 Cell 2。第一次会从官方来源下载 tracker 代码和权重。
3. Cell 3 选择一个视频，点击 **Load selected video**；支持 `.mov`、`.mp4`、`.m4v`，也接受大写扩展名。
4. Cell 4 设置清晰且五点可见的帧。默认步长为 8，因此帧号可用 0、8、16、80 等。严格按 `blunt-end → metal-intersection → sharp-end → metal-handle-intersection → handle` 点击，按 Done 确认。
5. Cell 5 运行跟踪。先去除检测到的黑边，再在跟踪坐标下预测，最后转换到保存的 640×480 图像坐标。预览最多显示 24 个均匀抽取的帧。
6. 检查漂移、关键点顺序和裁剪，把错误帧号写入 Cell 6 的 `BAD`。运行 Cell 6 后，少于五个可见关键点的帧和 `BAD` 中的帧会连同标签移到本地 `OUT/rejected/`，不进入 ZIP。需要恢复时，可从 rejected 下复制成对文件回原目录，再重新 review。
7. Cell 7 导出当前视频的 ZIP 到 Drive，再回 Cell 3 处理下一个。ZIP 名包含完整 clip 名；不会把所有视频合成一个包。

Cell 6 完成只表示执行了筛选；人工判断标注是否正确仍由操作者负责。上传前应检查 ZIP 里的所有图片，而非只看预览抽样。

## 输出与重跑

- 导出结构：`train/images/`、`train/labels/`、`data.yaml`。
- 每个目标的标签有 20 个字段，类别为 `0: rock_pick`，关键点形状为 `[5, 3]`。
- 代码默认跟踪/保存每第 8 帧，不代表跟踪了每一帧。
- 重跑同一个 clip 时，使用新的 `OUT`，避免旧帧残留；已有 ZIP 不会被覆盖。需要不同版本时，使用新的 `ANNOTATED_DIR`。
- Drive 原始视频只读取和复制，不会修改。中间结果和 rejected 目录在 Colab 临时磁盘上；结束会话前导出 ZIP，需要保留的 rejected 文件需另存 Drive。

## 显存不足

先缩短视频，或把 `TRACK_STRIDE` 从 8 增大到 16，并把 `EVERY` 调整为其整数倍。`CLICK_FRAME` 也必须是该步长的整数倍。必要时重启 Colab，再运行 1–2。增大步长会减少时间采样，需重新检查快速运动下的跟踪质量。原注释建议减小步长到 4，但这会加载更多帧，已修正。

## 交给 Roboflow / YOLO 训练之前

1. 将 ZIP 作为待审查标注批次导入，核对类别和五点顺序，不要把 tracker 预测直接视为人工真值。
2. 以原始录像或拍摄 session 划分 train/val/test。同一录像的相邻帧、切片及其近似副本应留在同一 split。
3. ZIP 中的 `train/` 只是暂存目录。完成独立划分后创建 `val/images` 及配套标签，或者使用平台重新导出的数据配置；原始 ZIP 不能直接作为完整训练集启动训练。
4. 保存数据版本、划分清单、YOLO 模型和软件版本、训练参数与 seed。当前来源没有这些训练记录，仓库不补造它们。
5. 用独立测试集评估检测框与关键点定位，另外检查遮挡、光照和相机变化。当前默认五点全可见筛选会排除遮挡样本。

格式参考：[Ultralytics Pose datasets](https://docs.ultralytics.com/datasets/pose/)。Tracker 参考：[CoTracker](https://github.com/facebookresearch/co-tracker)。
