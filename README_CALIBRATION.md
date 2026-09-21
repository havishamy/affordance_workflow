# 本次手眼标定的解释与服务器交接

依据是用户提供的原始结果和在另一台机器上实际使用的 launch。本机 `/home/dsj/ur5e` 下的同名 launch 不用于判断本次标定模式。

## 结论

- `eye_on_hand: true`：眼在手上。
- `robot_base_frame: base`：与本实验约定的 `world = UR base` 一致。
- `robot_effector_frame: tool0`：UR 控制器的全零 TCP 坐标；本地选择 `server.ee_frame: flange`，从当前 TCP 位姿去掉配置的 TCP offset 得到它。不要另加 ROS `flange` link 到 `tool0` 的旋转。
- 依据实际 ArUco 参数和标准 `aruco_ros single` 实现，数值对应 OpenCV 彩色光学坐标到 tool0 的变换，可作为服务器 `T_ee_cam`，不求逆、不额外补 `camera_color_frame` 到 optical 的旋转。

UR 对 tool0 的定义见 [官方坐标系说明](https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/rolling/doc/ur_description/doc/robot_frames.html)。

## 为什么 frame 名没有 optical，却按光学坐标使用

实际 launch 把 `camera_frame` 和 `reference_frame` 都设为 `camera_color_frame`，并把 RealSense 彩色图像和内参交给 ArUco。标准 ArUco 节点仅在这两个参数不同的时候执行 camera-to-reference TF 查询；相同时该变换为单位矩阵，检测数值仍在图像的光学坐标中，只被标记成了 `camera_color_frame`。

这是基于配置与[标准节点源码](https://github.com/pal-robotics/aruco_ros/blob/noetic-devel/aruco_ros/src/simple_single.cpp#L196-L206)的推断，并非验证了另一台机器的二进制版本。该结论适用于未修改坐标行为的标准 ArUco 节点、通常的 RealSense 彩色 CameraInfo；若改过节点或使用非零投影平移，应重新核对数值坐标。

该命名配置不适合直接按普通 RealSense TF 语义发布结果，否则容易产生坐标轴误用。这里仅解释既有结果供非 ROS 服务器使用，未修改原标定工程。

## 已生成文件

- `config/hand_eye_source.yaml`：原始四元数、平移、标定参数，以及实际 launch 中的 tracker 参数。
- `config/T_ee_cam.json`：服务器需要的矩阵，`status: confirmed` 表示此次方向及坐标解释已完成；`physical_accuracy_validated: false` 表示尚未完成独立实物精度检验。
- `config/hand_eye.yaml`：当前 TCP offset 为零时的本地矩阵，保留 `validated: false`，因此允许采集/定位/规划，仍阻止 `--execute`。

`T_ee_cam` 定义为：

```text
p_tool0 = T_ee_cam × p_opencv_camera
T_world_ee = T_base_tcp × inverse(T_tool0_tcp)
p_base = T_world_ee × T_ee_cam × p_opencv_camera
```

SAM3D 到 OpenCV 的翻转由服务器处理，本地不再次应用。

本地 TCP 经实机调整后设置为 tool0 到夹持中心沿法兰工具指向轴 `+Z` 方向 `0.140 m`。服务器的 tool0 标定保持不变；本地 `hand_eye.yaml` 已更新为 `inverse(T_tool0_tcp) × T_ee_cam`。运行前应在示教器确认蓝色 `+Z` 箭头确实从法兰指向夹持中心。

## 下一步

把 `config/T_ee_cam.json` 复制到服务器的一个新路径，并令服务器启动环境中的 `LAG_EYE_IN_HAND_CALIBRATION` 指向它。文件仅在本地生成，本次没有上传或重启服务器。服务器无需修改代码；若已有 API 进程，需在空闲时按其启动流程重新加载标定。

建立 SSH 隧道、设置本地 `LAG_API_TOKEN` 后，运行：

```bash
conda activate ur_data
cd /home/dsj/vla/affordance
python run_titration.py
```

先检查 `captures/<编号>/server_result/result.json` 和 ZIP 的中心 overlay/PLY。用可独立测得基坐标的固定标志点、多个静止拍摄位姿检验坐标误差；未通过检验前保留本地 `validated: false`。本次未测量夹持、尖端偏移或标定误差。

原 launch 将 `/camera/color/image_raw` 标为 `image_is_rectified: true`；这是否符合标定时的图像畸变情况，现有资料不能证明。若误差检验不通过，应同时核对内参、畸变处理、实际 marker 边长是否为 0.1 m、机器人运动学参数及相机安装是否保持不变。
