# AM970 D02 KL + ByteTrack 独立验证程序

此目录是正式源码位置，复用已验证的 D02 NPU 检测前后处理。当前用于独立开发板的限时摄像头验证，不是 ROS2 节点。训练、公司工程和原模型交付目录不受影响。

## 文件与依赖

- `detect_track.cpp`：V4L2 最新帧采集、NPU 推理、解码/NMS、逐帧跟踪、预览与记录。
- `tracking.hpp`：八类别关联隔离、参数读取、轨迹状态与 JSON 输出。
- `third_party/bytetrack/`：官方 ByteTrack C++ Kalman、LAPJV 和两阶段关联核心；来源、固定提交、修改说明及 MIT 许可随源码保存。
- `tracker.conf`：依次为低分阈值、高分阈值、新建阈值、第一阶段匹配代价上限、丢失保留毫秒数。
- `tracker_check.cpp`：真实跟踪核心的确定性状态检查，以及已记录检测序列的回放入口。
- `build.py`：使用现有 WSL SDK 交叉编译；`preview.py`：板端局域网预览。

依赖现有 AM970 VKNN SDK、OpenCV、Eigen 和 Buildroot AArch64 工具链，不依赖 NCNN，也不在板端运行 PyTorch。

## 模型与参数

使用 `/root/d02_detect_static_20261103/delivery_D02_KL/network_binary.nb`，SHA256：

`ac7690b71345ab579ec6da1953ecc6cd8b91230d00a44412e1c81a422f09e916`

类别：0 person、1 cat、2 dog、3 frisbee、4 sports ball、5 teddy bear、6 garden hose、7 rock。

模型输入保持 BGR UINT8、640×640 letterbox、补边 114；预处理归一化已在模型中。输出反量化为 12×8400，先解码并执行分类别 NMS（IoU 0.7、最多 300 框），再恢复原图坐标。摄像头实际为 640×480 MJPEG，申请采集 30 FPS，推理只消费最新帧。

`tracker.conf` 初始值：

```text
0.10 0.25 0.35 0.80 1000
```

低于 0.10 的框不进入跟踪；0.10–0.25 的框用于第二阶段续接，不能新建轨迹。高分阶段为 ≥0.25，新建要求 ≥0.35。第一阶段匹配参数是 **1−IoU 代价上限**，不是要求 IoU≥0.8。第二阶段代价上限沿用官方 0.5，未确认轨迹匹配沿用 0.7。该 C++ 实现不做检测分数融合，不能直接将参数与其他版本逐项等同。

每类独立关联，ID 在单次进程内全局递增，重启后重新计数。类别抖动可能造成断轨，当前没有跨类别纠错、ReID 或摄像头运动补偿。

## 状态与时间约定

| 状态 | 输出约定 |
| --- | --- |
| new | 首次高分新建，尚未确认；包括启动第一帧，统一要求后续高分关联确认 |
| tracked | 已确认且当前帧关联成功；可以由低分检测续接 |
| lost | 当前帧未观测到，`observed=false`、`confidence=null`；框为预测值，不得当作本帧检测 |
| removed | 超时、未确认失配、重复轨迹剔除或流结束后移除；只输出一次 |

每帧含类别、ID、状态、置信度、原图像素坐标 `xyxy` 和最后观测时间。Kalman 输出框可能越过原图边缘，预览绘制时裁剪；下游需按图像尺寸裁剪，不能把预测框当作原始检测框。原始 NMS 检测框在 `boxes` 和 `detections.txt` 中，已裁剪到原图。

保留 V4L2 原始采集时间戳、时钟类型和序号。设备提供单调时钟时，用采集时间计算轨迹保留期限；否则使用处理时单调时钟，原始采集时间仍单独保存。超出 1000 ms 的轨迹在关联前移除，避免断流后恢复旧 ID。Kalman 仍按每次处理帧一步预测，未引入变步长运动模型，快速运动、剧烈转动和低帧率仍可能断轨。

SIGINT/SIGTERM、限时结束及异常退出会输出停止/错误状态和终止事件；不发布旧轨迹为新观测。SIGKILL/掉电无法写终止事件，消费者必须检查帧号及超时；预览在三秒无更新时显示停止。

## 编译与板端运行

Windows PowerShell：

```powershell
wsl -d Ubuntu-22.04 -- python3 /mnt/e/cpp_project/lawn_mowing_robot_yolov8/mower_obstrast_check/src/deploy/am970/build.py
```

将 `build/detectTrack`、`build/tracker_check`、`tracker.conf`、`preview.py` 放入板端 `/root/d02_bytetrack/`。该目录的 `network_binary.nb` 指向上述固定模型。

板端先运行自动检查：

```bash
cd /root/d02_bytetrack
export LD_LIBRARY_PATH=/root/yolov8n_detect_npu_20260706/lib
./tracker_check
```

每次选一个新的输出目录，避免覆盖此前记录；以下示例录两分钟：

```bash
cd /root/d02_bytetrack
export LD_LIBRARY_PATH=/root/yolov8n_detect_npu_20260706/lib
mkdir -p /dev/shm/d02_bytetrack_manual/frames
./detectTrack --camera /dev/video0 /dev/shm/d02_bytetrack_manual 120
```

另一个终端启动预览（已有 8088 服务时，先停掉对应的旧预览进程）：

```bash
python3 /root/d02_bytetrack/preview.py /dev/shm/d02_bytetrack_manual
```

浏览器访问 `http://192.168.8.11:8088`。摄像头运行支持 1–600 秒，自动停止；Ctrl+C 也会正常结束。每三帧刷新预览，跟踪本身每个实际推理帧都运行。前 30 秒保存每三帧的原图和跟踪图，全程保存逐帧检测及轨迹，避免持续填满板端闪存。`/dev/shm` 断电丢失，验证结束后及时拉取文件。

固定序列回放：

```bash
./tracker_check /dev/shm/d02_bytetrack_manual/detections.txt > replay_tracks.jsonl
```

回放使用真实记录的检测、顺序和时间戳，验证跟踪结果可重现；不代替重新执行 NPU，也不代替带人工身份标注的跟踪精度评估。

## 验收边界

### ReID 对照实验（不等于实时上板部署）

`reid_features.py` 用主机 OpenVINO CPU 提取人员外观特征，输入为固定视频与板端记录的检测框；`tracker_check features.txt --features` 用板端真实 C++ 跟踪器重放带特征的检测。无特征输入继续使用原 ByteTrack。`evaluate_reid.py` 使用官方 TrackEval 对两组输出执行同口径评估。

实验仅对分数≥0.25、宽≥24、高≥64 像素的 person 框提取 256 维特征；不删除其他检测。第一阶段及未确认轨迹关联加入余弦外观距离，门限0.4、与IoU代价各占0.5，外观历史EMA为0.9。第二阶段低分关联与NMS保持原样。记录及结果见 `../../../docs/STORY34635_ByteTrack接入与验证/reid_trial/`。该参数是单段视频初步实验，不是推荐生产配置。

现有 `--camera`、`--video` 入口尚未调用ReID模型，实时运行仍为原ByteTrack；本轮通过带特征文件的离线重放验证关联逻辑。后续要部署外观提取模型并测量板端端到端开销。

### 公开视频实时预览

```bash
mkdir -p /dev/shm/d02_mot17_video/frames
cd /root/d02_bytetrack
LD_LIBRARY_PATH=/root/yolov8n_detect_npu_20260706/lib ./detectTrack --video /dev/shm/MOT17-09-SDP-raw.mp4 /dev/shm/d02_mot17_video 600
```

视频模式按顺序解码每一帧，不跳帧；使用视频帧号/FPS构造的媒体时间进行跟踪，不用较慢的处理耗时计算丢失期限。每遍结束记录 `loop_end`，清空轨迹后重新播放，ID 在进程内继续递增。页面及图像标明轮次，跨轮次 ID 不用于连续性评价。十分钟后自动停止。构建新增依赖板端已有 OpenCV videoio。

新增 `source`、`loop`、`tracking_clock` 字段；当前输出统一使用 `last_observed_ms`：摄像头模式为单调时间，视频模式为媒体时间。此前归档版本的 `last_observed_monotonic_ms` 字段保留在旧记录中，不改写历史结果。视频逐帧播放会随实际推理速度变慢，因此观看速度不代表原视频帧率。

自动状态检查用于验证低分续接、类别隔离、确认/丢失/恢复/移除及时间规则。摄像头实测用于观察真实检测对轨迹的影响。未经身份标注的新增 ID 数不能写成 ID Switch；同一 ID 恢复也不能单凭日志证明重关联到了同一个物理目标。

实测报告和原始证据见 `../../../docs/STORY34635_ByteTrack接入与验证/`。ROS2 消息、深度关联、实车与长期稳定性验收另行开展。
