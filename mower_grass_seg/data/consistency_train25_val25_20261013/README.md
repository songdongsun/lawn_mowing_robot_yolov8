# 草坪分割模型三端一致性测试

2026-10-13，针对 STORY #25028 的第 6 项，对指定目录的全部 50 张图片完成 PyTorch、ONNX Runtime 和 AM970 板端 NPU 比较。现有混合精度模型保持不变，未替换摄像头或静态部署程序。

## 测试结论

| 比较 | 图片数 | 平均草坪 Mask IoU | 最低 Mask IoU | 平均 Dice | 检测数量一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch 与 ONNX | 50 | 100.00% | 100.00% | 100.00% | 50/50 |
| PyTorch 与 NPU | 50 | 97.85% | 87.65% | 98.90% | 48/50 |
| ONNX 与 NPU | 50 | 97.85% | 87.65% | 98.90% | 48/50 |

NPU 相对 PyTorch 的训练集 25 张平均 IoU 为 97.90%，验证集 25 张为 97.80%。全体像素累计交并比为 98.07%，与逐图平均值是不同统计口径。没有两端同时输出空 Mask 的样本。

这是输出一致性测试，不是与人工标注比较的准确率，也不替代独立外场测试。尚未约定允许误差，因此不作验收通过的结论。IoU 按每张图片所有 grass 实例的并集计算，不是逐实例匹配的 mAP。

## 重点复核图片

| 图片编号 | PyTorch 与 NPU Mask IoU | 说明 |
| --- | ---: | --- |
| 044 | 87.65% | 最低分；NPU 保留的前景框更宽，左侧草坪覆盖范围不同 |
| 002 | 90.34% | 第二低分，需结合图片检查边界差异 |
| 001 | 93.62% | 建议复核 |
| 028 | 93.81% | 建议复核 |
| 018 | 94.14% | 建议复核 |

000 和 042 两张图片的检测数量从 PyTorch 的 2 个变为 NPU 的 3 个。框数差异不直接等同于草坪区域丢失或增加，需结合 Mask 查看。

## 测试方法

- 数据源为 `test_train25_val25_b07_v4`，由 manifest.csv 确认训练集 25 张、验证集 25 张。
- 同一原图生成同一份 640x640 letterbox 图，填充值 114；使用无损 PNG 传到板端。
- PyTorch 和 ONNX 输入为 RGB float32 /255；板端输入为 BGR UINT8 HWC，由已有 NB 前处理节点完成转换。
- NPU 使用现有 UINT8 与 FP16 混合精度 NB，导出两个 FP32 输出：37x8400 和 32x160x160。
- 独立测试工具直接复用 `main_mower_static.cpp` 中的 `vknn_run`，逐图保存实际板端输出，不使用软件假量化替代。
- 三端原始输出在主机使用同一后处理：置信度 >=0.25、框 NMS IoU >0.45 时抑制、Mask >0.5；还原 letterbox，裁剪至检测框，合并 grass 区域。
- 后处理沿用部署逻辑，但这是统一主机解码的转换一致性测试，不是板端 C++ 后处理逐位一致性的验证。相同置信度候选的排序可能与 C++ std::sort 不同。
- 模型、50 张输入和 100 个输出文件的 SHA256 均与板端记录核对通过。

原始浮点输出并非完全相同；PyTorch 与 ONNX 的最大置信度绝对差约为 1.10e-5，但本批图片最终二值 Mask 一致。

## 耗时

50 次板端日志中 inference 字段平均 36.05 ms，范围 35.845 至 36.6404 ms。该字段包含 VKNN 输入提交、推理和输出获取，不是硬件内部计时；不包括 Mask 后处理，也不是摄像头端到端帧耗时。测试工具每张图重新初始化模型，不能直接用此批测试计算连续摄像头 FPS。

## 文件索引

- `comparisons/000.jpg` 至 `049.jpg`：从左至右为原图、PyTorch、ONNX、NPU，红色是草坪 Mask；标题栏占用顶部显示区域，原尺寸 Mask 另存。
- `masks/`：原图尺寸无损二值 Mask，白色为草坪。
- `per_image.csv`：三组比较的 150 行逐图指标及原文件名。
- `summary.json`：汇总、分组指标、模型及原图校验值、样本编号映射。
- `reference/*_detections.json`：三端置信度、框坐标和候选数量。
- `board_run.log`、`board_checksums.txt`：实际板端运行记录及校验值。
- `versions.json`：本地 PyTorch、ONNX Runtime、OpenCV 等版本。

原始张量、准备好的输入和测试脚本保存在本地 am970 工程。复算命令从 `/root/cpp_project` 执行：

```bash
am970/model/tools/conda-acuity310/bin/python am970/model/scripts/compare_mower_backends.py \
  --data /mnt/e/cpp_project/lawn_mowing_robot_yolov8/mower_grass_seg/data/test_train25_val25_b07_v4 \
  --out am970/model/results/consistency_train25_val25_20261013 --npu-only
```

后续需要确定平均和单图允许误差，并结合独立外场人工标注验证实际分割质量。最低分样本可能包含参考模型自身的偏差，不能只凭两端差异判定哪端更准确。
