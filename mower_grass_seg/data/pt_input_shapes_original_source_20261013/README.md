# B07_v4 训练配置与部署输入复核

## 权重来源

当前部署源模型 mower_yolov8n_seg.pt 与项目 runs/segment/B07_v4/weights/best.pt 的 SHA256 完全相同：fded736580dabb15a9da164f0bae13d053e0e2190ed6520924688b870f4a9163。

检查了 B07_v4/args.yaml、checkpoint 内 train_args 和 results.csv，而非直接套用当前 train_seg.py。当前训练脚本已经指向 B08，mosaic=0.0，不代表 B07_v4 当时参数。

## 实际训练配置

| 参数 | 值 |
| --- | --- |
| imgsz / rect / multi_scale | 640 / false / 0.0 |
| batch / optimizer / lr0 | 16 / AdamW / 0.001 |
| epochs / patience | 200 / 20 |
| mosaic / close_mosaic | 0.3 / 80 |
| translate / scale | 0.1 / 0.5 |
| fliplr / flipud | 0.5 / 0.0 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 |
| degrees / shear / perspective | 0 / 0 / 0 |
| mixup / cutmix / copy_paste | 0 / 0 / 0 |
| mask_ratio / overlap_mask | 4 / true |

训练是方形 640x640 张量，并有平移、缩放、颜色和水平翻转等增强。配置中的 augment=false 是推理增强开关，不能解读为训练没有数据增强。auto_augment 和 erasing 位于分类增强路径，不能把 args.yaml 中这些通用字段算作已启用的分割增强。

results.csv 共 128 轮，最佳指标对应第 108 轮，与 checkpoint 内保存指标相符。close_mosaic=80 对计划 200 轮意味着约第 121 轮开始关闭 Mosaic。因此当前 best.pt 来自关闭 Mosaic 之前，不能把它当成最后 80 轮关闭 Mosaic 后的模型。这是后续实验线索，不是已证明的误检原因。

## 验证与预测预处理

项目源码中 detect/train.py 对验证显式使用 rect=True，矩形批次按比例、stride、pad 计算实际尺寸，不等同于固定 640x640。model.val 默认也开启 rect。不要将训练 args 中 rect=false 自动套用到验证。

单张 model.predict 默认 rect=True，项目 predictor.py 据此做最小补边。在本批 640x480 原图上，实际输入为宽 640、高 480。

原 B07_v4_test50_eval 脚本未固定方形输入，其 imgsz=640/conf=0.25 的逐图 TP、FP、FN、TN 与本次 640x480 复核全部一致。原报告的 grass_iou=98.7953% 是累计像素交并比，不是逐图平均 IoU。

部署 NB 则固定 640x640，原图上下各补 80 行、填充值 114。虽然训练使用方形张量，历史验证/预测并没有以完全相同的固定方形条件覆盖部署行为；这不等于 letterbox 算法错误。

## 原项目源码复跑

checkpoint 与历史评估记录是 Ultralytics 8.4.128，转换环境是 8.3.0。本次通过 PYTHONPATH 加载原项目的 8.4.128 源码，保持本地 PyTorch 2.2.2 CPU 运行，使用标准后处理，conf=0.25、NMS IoU=0.7、retina_masks=True。不是完整重建原训练 GPU 环境，历史评估记录的 PyTorch 为 2.14.0+cpu。

实际 RGB /255 输入张量的形状和值逐次核对。与之前 8.3.0、NMS=0.45 的对照指标相同，因此本批误检不能归因于该版本差异或这两个 NMS 阈值差异。

| 对当前标注的指标 | 宽640高480 | 640x640补边 |
| --- | ---: | ---: |
| 50 张平均 Mask IoU | 98.72% | 97.18% |
| 验证来源 25 张平均 IoU | 98.70% | 97.97% |
| 018 IoU | 98.49% | 47.52% |
| 总误检像素 | 59,534 | 120,581 |
| 总漏检像素 | 24,877 | 48,964 |

018 的大面积水泥路误检仍在固定方形标准 PT 推理中复现，不是新增 FP16 配置导致。034 还存在裸土附近误分，009 有裸土/草坪边界外溢。其他样本的红色区域不能全部叫道路误检，例如 022、034 的远处区域可能是未标注草坪，需要人工复核标注覆盖规则。

## 文件与下一步

comparisons 中左侧为当前标注，中间为640x480预测，右侧为640x640预测；预测绿色为TP、红色为FP、蓝色为FN。masks 保存原尺寸二值图，per_image.csv 为逐图指标，summary.json 保存版本与数据哈希。

建议在确定部署固定方形的前提下，先补齐固定640x640的独立验证，再针对道路/裸土误检样本复核标注并做受控微调。关闭Mosaic、降低几何增强是否改善，需要独立实验，不能仅根据本次观察断言。也不要把模型间98.89%一致性当作相对标注的准确率。

本次没有修改训练数据、训练配置、权重或板端默认程序；没有重新训练，也没有转换640x480 NB。
