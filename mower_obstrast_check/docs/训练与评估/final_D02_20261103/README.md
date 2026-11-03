# 障碍物检测第一版 D02 使用说明

推荐权重为best.pt，第194轮，8类，YOLOv8n三尺度检测头加P3通道注意力。输入640，初始conf0.25、NMS IoU0.7、max_det300。详细结果见[交付报告](../障碍物检测第一版模型评估与选型交付报告_20261103.md)。

[Word版交付报告](模型评估与选型交付报告.docx)（含12张表格和4张嵌入图片）。

## 运行

在本项目根目录，使用能导入本地ultralytics的Python环境运行。Windows已验证环境为D:/itsoft/conda/envs/grass_seg/python.exe。模型包含C2fChannelAttention，不能假设原版PyPI包能直接加载；保留当前项目代码。环境及仓库版本见environment.json，模型内原始train_args见evaluation/D02/full/checkpoint.json。CPU FP32为本次已验证配置。

```python
from pathlib import Path
import yaml
from ultralytics import YOLO

cfg = yaml.safe_load(Path("mower_obstrast_check/docs/训练与评估/final_D02_20261103/predict.yaml").read_text())
model = YOLO(cfg.pop("model"))
results = model.predict(source="你的图片或图片目录", **cfg)
for result in results:
    for box in result.boxes:
        print(int(box.cls.item()), float(box.conf.item()), box.xyxy[0].tolist())
```

请替换source。predict.yaml中的路径相对项目根目录；移动交付目录后相应修改model/project。CUDA环境可把device改为实际GPU编号，但本次CPU计时不代表该设备性能。输出标注图片及YOLO文本保存到inference/predict或自动递增目录，文本附带置信度；Python接口的xyxy为原图像素坐标。

## 类别

0 person 人员；1 cat 猫；2 dog 狗；3 frisbee 飞盘；4 sports ball 运动球；5 teddy bear 玩具熊；6 garden hose 软管；7 rock 石块。

## 交付范围

training保留best来源的配置、last、CSV与完整日志；evaluation保留8个候选五套范围的指标及D02输入清单；failure_cases为失败对照；smoke保存本次8图推理输出。cpu_benchmark.json只代表CPU前向成本。SHA256SUMS.json覆盖交付文件，不包含它自身。

conf0.25是固定离线口径下的初始使用值，尚未针对真实双目场景优化。石块、小球及远处人员漏检和软管误检仍存在。板端转换、量化、ROS2以及实车验收属于后续任务。
