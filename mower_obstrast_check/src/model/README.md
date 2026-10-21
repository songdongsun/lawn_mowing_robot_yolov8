# 障碍物检测训练入口

参照草坪分割训练入口，使用 YOLOv8 检测预训练权重和当前 12 类数据。此处提供代码，尚未启动冒烟或正式训练。
从仓库根目录执行，Windows/Linux 均可；使用已安装本仓库 Ultralytics、PyTorch 和 PyYAML 的环境。

## 第一次：先检查配置，再冒烟

```bash
python mower_obstrast_check/src/model/train_detect.py --smoke --device 0 --dry-run
python mower_obstrast_check/src/model/train_detect.py --smoke --device 0
```

把 `0` 换成允许使用的 GPU 编号；CPU 可写 `--device cpu`。不会自动选择或占用其他服务器。
默认加载 `yolov8n.pt`，本地不存在时 Ultralytics 可能下载；离线环境使用 `--model /path/to/yolov8n.pt`。

冒烟默认：1 轮、640 输入、batch=2、workers=0、AMP 关闭、训练集 fraction=0.002（当前约 133 张）、
关闭 Mosaic 和绘图，输出到 `mower_obstrast_check/runs/detect/smoke`。
保留验证步骤，仍会评估完整验证集 2,840 张，所以不是仅运行几个 batch。
训练子集不保证涵盖全部少数类，冒烟仅验证加载、前反向、验证和权重保存，不证明 12 类训练效果。
`--set val=false` 也不能保证完全跳过末轮验证，框架在最后一轮仍可能验证。

## 参数全部通过命令调整

```bash
# 冒烟参数也可以显式覆盖；本命令仍仅 1 轮
python mower_obstrast_check/src/model/train_detect.py --smoke --device 2 --batch 4 --fraction 0.005 --name smoke_gpu2

# 正式训练示例：冒烟通过后再决定是否执行
python mower_obstrast_check/src/model/train_detect.py --device 2 --name D01 --epochs 200 --patience 20 --imgsz 640 --batch 16 --workers 4 --optimizer AdamW --lr0 0.001 --mosaic 0.3 --close-mosaic 10 --seed 0

# 其他框架参数用 --set；布尔值写 true/false
python mower_obstrast_check/src/model/train_detect.py --smoke --device 0 --set amp=true cache=false fliplr=0.5 --dry-run
```

优先级：普通默认值 → 冒烟预设 → 显式 `--参数` → `--set KEY=VALUE`。
因此显式 `--epochs 5` 会将冒烟改成 5 轮。`--set` 支持框架训练参数，包括 `batch=0.5`、
`resume=true` 等；未知参数由 Ultralytics 校验。续训应同时通过 `--model` 指定对应 `last.pt`，
框架可能恢复原运行参数，不应将其当作一次全新实验。不要使用 `exist_ok=true` 覆盖既有实验。
`--help` 查看常用参数；`--dry-run` 只读取并打印配置，不加载模型、不下载、不写文件，不验证 GPU 或框架参数兼容性。

## Windows / Linux 数据路径

默认读取项目内 `data/yolo_check/data.yaml`，将其数据根目录解析为这台机器上的实际目录，
避免服务器沿用 YAML 中的 `E:/...`。真正训练时，在 `logs/train_configs/` 保存独立配置副本，
原始 YAML、图片、标签及 train/val 划分不变。

数据放在其他位置时：

```bash
python mower_obstrast_check/src/model/train_detect.py --smoke --device 0 --data-root /mnt/datasets/yolo_check
```

如果另有配置文件，用 `--data /path/to/data.yaml`，其路径设置保持原样；需要重定位时再附加 `--data-root`。
`--data-root` 更改 YAML 的根目录，不会改写其 train/val 字段中的绝对路径。
默认输出实验目录由框架自动递增，训练配置和 best/last 权重保存在对应运行目录。
当前数据不平衡和少数类验证样本不足仍然存在，正式评估须逐类查看，不能只看总体 mAP。
