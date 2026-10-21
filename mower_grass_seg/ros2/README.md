# STORY #25034 草坪分割 ROS2 节点

本包使用 AM970 VKNN 模型 `network_binary.nb`，不是 CPU/ONNX 备选推理。输入接受 `bgr8` 或 `rgb8`，输出 `mono8` 草坪 Mask（草坪 255，非草坪 0），分辨率与输入一致，`header.stamp` 和 `header.frame_id` 原样传递。CameraInfo 用于检查分辨率；模型不做去畸变，因此 Mask 坐标属于输入图像的像素坐标系。若相机输出未经校正，后续消费者不得把它当成校正图坐标。

后处理沿用 [既有验证脚本](../deliverables/STORY25028_B08_v3_2_20261014/09_scripts/compare_mower_backends.py) 的 B08_v3_2 规则：640 letterbox、114 填充，置信度 0.25，NMS IoU 0.45，原图尺度概率阈值 0.5，实例框外裁剪后取并集。模型应为 SHA256 `9f162de626472adaac621eb5ffde0fff4dc3e7daf6089ec1810c4141023bf898`，输出必须为 `[1,37,8400]` 和 `[1,32,160,160]`，否则节点拒绝启动。

## 板端构建和运行

需要板端 ROS2 C++ 环境、OpenCV、与板端架构匹配的 VKNN SDK（`include/Vknn.h`、`lib/libvknn.so`），并确保动态库运行时可找到。将本目录下的 `mower_grass_seg` 放入 ROS2 workspace 的 `src/`，然后在板端执行（以下路径按实际 SDK 安装位置替换）：

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --packages-select mower_grass_seg --cmake-args -DVKNN_SDK_ROOT=/path/to/vknn/sdk
source install/setup.bash
export LD_LIBRARY_PATH=/path/to/vknn/sdk/lib:/usr/lib/amicro_ndk:${LD_LIBRARY_PATH}
sha256sum /root/deploy_mower_cam_vknn/model/network_binary.nb
ros2 launch mower_grass_seg grass_seg.launch.py image_topic:=/实际图像话题 camera_info_topic:=/实际相机信息话题
```

模型路径、置信度、NMS、Mask 阈值和统计窗口见 `mower_grass_seg/config/grass_seg.yaml`；Mask 默认 `/grass/mask`。单线程回调和 SensorDataQoS depth=1 避免应用层推理排队；下游若使用可靠 QoS，须检查是否能与最佳努力发布端匹配。不要同时让旧相机独占采集进程与 ROS2 相机驱动争用同一设备。

## 上机验收记录

当前尚未在目标机器执行此 ROS2 节点，以下指标不能用旧独立相机程序的耗时替代。先确认芯片型号、ROS2 发行版、相机话题、CameraInfo 话题、模型路径和 SDK 路径，并检查相机图像编码。启动后录制至少 30 分钟的稳定运行日志和室外草坪/非草坪样例：

```bash
ros2 topic info /grass/mask -v
ros2 topic hz /grass/mask
ros2 topic echo /grass/mask --field header --once
ros2 topic echo /实际图像话题 --field header --once
```

日志每 100 帧给出发布帧率、P50/P95 节点处理耗时（收到图像回调至发布完成）、进程 RSS；记录起止 RSS、丢帧、NPU 错误及重启次数。此耗时不含相机采集前的延迟；若需端到端传感器延迟，需相机与板端时间同步后比较采集时间戳和发布时间。人工核对输出图像尺寸、非空草坪、障碍物/道路不被并入草坪，以及输入输出同帧 `stamp/frame_id`。报告只填写实测数值，不用历史 benchmark 冒充本次验收。
