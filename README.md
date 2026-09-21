# 眼在手上采集 → 服务器定位 → UR 滴管实验（无 ROS）

`run_titration.py` 使用服务器 `/v1/titration/jobs` 异步 API：采集并保存四个文件 → 上传 → 轮询 → 读取胶头和杯口世界坐标 → 下载结果 ZIP → 生成机械臂路径。默认只采集、定位和规划；加 `--execute` 才执行夹爪与机械臂动作。

实现与默认输出都在 `affordance/` 内，无需修改 `ur_data_collection` 或 `lerobot`。服务器接口和输入格式分别见 `README_LOCAL_SERVER_API.md`、`README_LOCAL_CAPTURE_AND_TRANSFER.md`。

## 1. 坐标系与配置

本实验已确认 **world 就是 UR base**。API 返回的 `dropper_bulb_center_world`、`beaker_mouth_center_world` 直接作为基坐标米制点，客户端不会再乘手眼矩阵，也不会再次应用 SAM3D 坐标翻转。

修改 `config/config.yaml` 中：

- `robot.ip`、`robot.tcp_offset`、负载与重心、速度和加速度。
- `camera.serial`、分辨率，以及 `gripper.port`。
- `server.url`：SSH 隧道地址，例如 `http://127.0.0.1:8000`，不要追加任务路径。
- `server.ee_frame`：已依据实际标定的 `tool0` 设置为 `flange`。此处指控制器全零 TCP 坐标，不是 ROS 中另一个轴向不同的 `flange` link。

上传位姿按以下规则生成：

```text
ee_frame: tcp
  T_world_ee = T_base_tcp（getActualTCPPose 转换成 4×4 矩阵）

ee_frame: flange
  T_world_ee = T_base_tcp × inverse(T_flange_tcp)
  T_flange_tcp 来自 robot.tcp_offset
```

本次实际标定的来源、ArUco frame 名与光学坐标的解释，以及服务器交接步骤见 [README_CALIBRATION.md](README_CALIBRATION.md)。服务器标定方向要求 `p_ee = T_ee_cam × p_camera`。

`config/hand_eye.yaml` 已填入本次标定在当前零 TCP offset 下的矩阵，保留 `validated: false`，等待实物精度验证。此文件表示 **T_tcp_camera**；若修改 TCP offset，应更新为 `inverse(T_flange_tcp) × T_flange_camera`。确认独立测量误差及实际 TCP 后才设 `validated: true`。新 API 使用服务器上的标定，本地文件不会自动上传，也不会用于再次变换世界坐标；两端标定需人工核对，目前 API 未提供标定版本比对。

## 2. 环境与服务器连接

复用采集工程环境：

```bash
conda activate ur_data
cd /home/dsj/vla/affordance
python -m pip install -r requirements.txt
```

`pyDHgripper` 使用原采集工程中已能运行的同一个包。确认服务器已通过标定检查并启动，然后在独立终端保持隧道运行：

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@SERVER
```

在运行实验的终端设置：

```bash
export LAG_API_TOKEN='与服务器一致的密钥'
```

Token 只用于 HTTP 请求头，不写入采集记录。任务提交不自动重试，以免响应超时后重复排队；查询可重试暂时性错误。`server.job_timeout_s` 限制排队/计算等待时间（默认 1800 秒），`server.timeout_s` 是单次请求超时，`bundle_scope` 可设 `results` 或 `all`。

## 3. 先采集，再拿到两点

机械臂保持静止，确保胶头与杯口可见。只保存服务器需要的输入，无需设置 Token 或启动服务器：

```bash
python run_titration.py --capture-only
```

一次采集、上传、下载、打印两个基坐标与规划路径：

```bash
python run_titration.py
```

已经完成定位且滴管、烧杯、机器人底座均未移动时，可以复用保存的世界坐标，跳过相机和服务器。先按当前机器人位置和当前配置重新生成路径：

```bash
python run_titration.py --reuse-capture captures/CAPTURE_ID
```

检查该目录下新建的 `reuse_runs/<时间>/plan.json` 后执行：

```bash
python run_titration.py --execute-saved captures/CAPTURE_ID
```

复用模式读取 `server_result/result.json`，校验任务已完成、坐标为米制 base/world、结果与 metadata 一致。保存的是物体在 base 中的最终坐标，因此调整 TCP 或夹爪参数后仍可复用；程序会明确警告 TCP 变化，并按当前 TCP、当前位置和运动配置重新生成全部路径。它不会连接相机、访问服务器或复用旧 motion plan。每次尝试均写入独立的 `reuse_runs` 子目录，不覆盖原始定位结果。复用的前提仍是物体和机器人底座没有移动。

这两个模式不发送抓取运动或连接夹爪，但沿用现有 `URRobot.connect()`：会连接 RTDE 控制接口、设置 TCP 和负载，异常时可能发送 `stopL`。不要与其他正在控制机械臂的程序同时运行。

输出目录：

```text
captures/<capture_id>/
  rgb.png                  彩色 PNG
  depth.png                对齐到彩色图的 uint16 原始深度
  camera.json              fx/fy/cx/cy、尺寸、深度尺度与畸变信息
  T_world_ee.json           {"T_world_ee": [[...], ...]}，米制 4×4 矩阵
  capture_info.json         TCP offset、位姿、坐标系和采集时间信息
  server_result/
    job.json               最新任务状态及任务 ID
    result.json            API 原始两点坐标
    titration_<job_id>_results.zip
  metadata.json            两个基坐标、规划路点及采集信息
  failure.json             失败/中断时尽力保存错误
```

四个输入在上传前保存，因此服务器失败也不会丢失采集。`--capture-only` 只生成采集文件，不生成服务器结果或规划。上传深度尺度取 RealSense SDK 实际值，不固定为 `0.001`。

代码比较采集前后 TCP 位姿，拒绝超过位置容差或 0.02 rad 的变化。记录 RGB/深度的 SDK 时间戳及各自时间域，机器人位姿时间则是主机读数时间，二者不能直接假定为同一时钟，不是硬件同步保证。采集和等待期间需保持场景不变。执行前还会再次检查机器人位姿。

也可直接使用现有独立客户端重传已保存的采集，深度尺度读取 `capture_info.json`：

```bash
python client.py run \
  --server http://127.0.0.1:8000 \
  --rgb captures/CAPTURE_ID/rgb.png \
  --depth captures/CAPTURE_ID/depth.png \
  --camera captures/CAPTURE_ID/camera.json \
  --world-ee captures/CAPTURE_ID/T_world_ee.json \
  --depth-scale 0.001 \
  --capture-id CAPTURE_ID \
  --output-dir captures/CAPTURE_ID/server_result
```

上面的 `0.001` 仅为示例。此独立客户端不会控制机械臂，也不会将旧结果自动执行；实验入口每次使用新采集。

## 4. 两点如何变成实验动作

规划使用固定的 `tool_orientation_rotvec`，未推断滴管轴向或抓取姿态。滴管摆放与夹爪朝向必须匹配。参数均需在实际设备上测量：

```text
TCP 抓取位置 = 胶头中心 + grasp_offset_m
TCP 滴定位置 = 杯口中心 + beaker_offset_m + [0, 0, dispense_height_m]
```

这些偏移在 base 坐标中表达。`dispense_height_m` 是 TCP 相对杯口的高度，不是滴管尖端的净空。如果固定姿态下 TCP 到尖端的基坐标向量为 d，想让尖端位于杯口上方 h，则 TCP 目标应为 `杯口 + [0,0,h] - d`。应同时用携带滴管的完整尺寸核对 `transit_z_m` 与全部路径。

确认 `workspace_min/max`、夹持力、`open_position/grasp_position/squeeze_position`。当前示例 `350` 和 `0` 没有针对实际胶头验证。API 不提供置信度，程序记录为缺失，不伪造为 1.0；`min_confidence` 无法替代本次 overlay/PLY 和实物坐标检查。

先检查 `server_result/result.json`、ZIP 中的中心图和 overlay/PLY，以及 `metadata.json` 的全部路点。完成标定和参数验证后：

```bash
python run_titration.py --execute
```

程序重新采集和定位，下载结果、打印路径后停在 `EXECUTE` 确认处。检查本次结果后再输入。确认之前不会连接或初始化夹爪；`--yes` 显式跳过此确认。

动作顺序：打开夹爪 → 升高 → 胶头上方 → 下降夹持 → 提起 → 移至杯口上方 → 下降 → 挤压/恢复夹持 → 抬起。执行中连续运行，不具备逐动作暂停；最终继续夹持滴管。

当前只实现固定次数挤压，没有吸液、抓取成功检测、单滴体积控制或滴定终点判断。实验需明确预先装液方式，并先用低速和替代道具验证。笛卡尔边界检查不包含完整的障碍物、夹爪和滴管碰撞检测。

## 5. 查询失败任务与测试

超时只停止本地等待，不取消服务器任务。可用保存的任务 ID 查询或重下 ZIP：

```bash
python client.py status JOB_ID
python client.py download JOB_ID --output-dir captures/CAPTURE_ID/server_result
```

部分下载保留为 `.zip.part`；成功后才改名为 `.zip`。首次提交若超时且未取得任务 ID，应先检查服务器队列再重传。

无硬件测试：

```bash
python -B -m unittest discover -s tests -v
```

测试使用模拟 HTTP 响应与硬件，覆盖 multipart 编码、状态轮询、失败/超时、结果与 ZIP 保存、坐标系校验、TCP/法兰变换，以及执行前确认和位姿变化保护。它们不代表已完成真实服务器或机器人联调。
