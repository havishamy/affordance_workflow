# Affordance：UR 眼在手上化学实验客户端

本目录包含本地 RGB-D 采集、服务器 affordance 定位、UR 路径规划和可选真机执行。`world` 统一表示 UR `base`，默认仅采集/定位/规划；只有显式传入 `--execute`，且配置闸门和人工确认均通过后，程序才会控制机械臂或夹爪。

当前推荐入口是 `run_three_stage_demo.py`。文件名为了兼容历史命令仍叫 three-stage，实际已经包含四个阶段：拔塞、倾倒、滴液和玻璃棒搅拌。

## 1. 入口与适用范围

| 入口 | 用途 | 默认是否运动 |
|---|---|---|
| `run_three_stage_demo.py` | 推荐的四阶段流程；可重新观察或复用已有结果 | `--observe 0` 不运动；`--observe 1` 会移动到观察位；阶段动作还需 `--execute` |
| `run_titration.py` | 旧的两目标滴管/烧杯滴定流程 | 默认只采集、定位和规划；`--execute` 才执行 |
| `run_complex_demo.py` | 独立的早期四阶段实验/格式验证 | 默认规划；当前 `physical_accuracy_validated: false` 阻止真机执行 |
| `run_depth_bulb_grasp.py` | 原始深度胶头定位与可选夹持诊断 | 默认离线；`--live` 采集，`--observe` 会移动，`--live --execute` 抓取闭爪 |
| `client.py` | 已保存采集的 API 提交、查询和结果下载 | 不控制机器人、相机或夹爪 |

随时用 `python <入口> --help` 查看当前参数，参数定义以代码输出为准。

## 2. 环境与配置

```bash
conda activate ur_data
cd /home/dsj/vla/affordance
python -m pip install -r requirements.txt
```

`pyDHgripper` 需要复用 `ur_data_collection` 中已能工作的安装。主要配置：

- `config/config.yaml`：UR IP、TCP、负载、夹爪串口、相机、服务器和旧滴定运动参数；
- `config/three_stage_demo.yaml`：推荐四阶段流程、观察模式、各阶段运动和质量阈值；
- `config/complex_demo.yaml`：独立早期复杂流程；
- `config/T_ee_cam.json`、`config/hand_eye.yaml`、`config/hand_eye_source.yaml`：服务器和本地使用的手眼标定及来源记录。

真机前逐项核对，不要仅相信仓库里的历史值：

- `robot.ip`、`tcp_offset`、负载、重心、速度和加速度；
- `gripper.port`、开度、夹持力和速度；
- 相机序列号、分辨率、手眼标定和相机安装是否改变；
- 工作空间、观察位、转运高度、抓取开度、瓶塞停车位、滴管长度和杯内净空；
- 配置中的 `physical_execution_validated`、`stirring_motion.validated` 或类似字段是否确实对应本次现场验证。

这些 `validated: true` 只是操作员声明，不是程序自动测量结论。不同标定文件中的状态字段用途不同，也不能互相替代。

### 夹爪串口

当前实机约定 `0 = 完全打开`、`1000 = 完全闭合`。配置使用持久化的 `/dev/serial/by-id/...` 路径；若该路径未生成，先确认实际设备：

```bash
find /dev -maxdepth 3 \( -name 'ttyUSB*' -o -name 'ttyACM*' \) -ls
lsusb | grep -i -E 'FTDI|Future Technology'
ls -l /dev/serial/by-id/ 2>/dev/null
```

不要盲试其他串口。如果设备是 `/dev/ttyUSB0`，可以只为本次命令临时覆盖路径，不修改 YAML：

```bash
python -c 'from dataclasses import replace; from affordance.config import AppConfig; from affordance.hardware import DHGripper; cfg=AppConfig.load("config/config.yaml"); gc=replace(cfg.gripper, port="/dev/ttyUSB0"); g=DHGripper(gc); g.connect(); g.open(); g.disconnect(); print("夹爪已打开")'
```

执行前托住被夹物体、移开手指并确保下方无人。该命令只控制夹爪，不移动 UR；如果夹住人体，优先急停并按设备说明使用机械/断电释放。

## 3. 服务器连接与数据契约

推荐让服务器 API 只监听回环地址，并通过 SSH 隧道访问：

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@SERVER
```

实验终端设置与服务器一致的 token：

```bash
export LAG_API_TOKEN='与服务器一致的密钥'
```

不要把 token 写入仓库或采集目录。主要端点为：

| 路径 | 用途 |
|---|---|
| `/health` | 健康检查 |
| `/v1/titration/jobs` | 旧两目标滴定定位 |
| `/v1/complex-demo/jobs` | 广口瓶、烧杯和滴管的复杂场景定位 |
| `/v1/stirring/jobs` | 滴管归位后重新观察玻璃棒 |

任务接口均采用提交、轮询和下载模式：

```text
uploading -> queued -> running -> completed
                            \-> failed/interrupted
```

一次采集的核心输入：

```text
captures/<capture_id>/
  rgb.png             彩色图
  depth.png           对齐到彩色图的 uint16 原始深度
  camera.json         内参、尺寸、畸变和实际 depth scale
  T_world_ee.json      拍照时的 4x4 位姿
  capture_info.json    TCP、时间戳和坐标系记录
```

深度米制值为 `depth_uint16 * depth_scale_m_per_unit`，零深度无效。RGB、深度和相机内参尺寸必须一致。`T_world_ee` 的具体末端语义由 `server.ee_frame` 决定；当前流程会按配置从实际 TCP 位姿生成上传矩阵，服务器不应再次去除 TCP offset。

所有结果必须使用有限的米制 `world/base` 坐标。复杂接口返回模型 ID、pose/scale、世界点云和已有 affordance 区域点集；它不负责猜测液面、瓶口轴、滴管尖端、抓取姿态、倾倒量、滴数或机器人路径。模型 ID 和 canonical 坐标若不能跨运行稳定，本地几何先验就不能安全复用。

## 4. 推荐四阶段流程

### 4.1 只复用本地结果并规划

```bash
python run_three_stage_demo.py --observe 0
```

模式 `0` 优先读取 `captures/three_stage_latest_result.json` 指向的最近一次已通过基础检查的结果；按配置可回退到 YAML 中的 `capture_dir/result_dir`。它不连接相机或服务器，也不运动，除非另加 `--execute`。

### 4.2 重新观察、提交服务器并规划

```bash
export LAG_API_TOKEN='与服务器一致的密钥'
python run_three_stage_demo.py --observe 1
```

模式 `1` 会连接 UR，移动到配置的参考观察位，采集 RGB-D，提交 `/v1/complex-demo/jobs`，下载并检查结果，然后规划。即使没有 `--execute`，观察移动也是真实机械臂运动。

初始画面必须同时包含瓶塞、瓶身、烧杯口和胶头。瓶塞已经移走后不要重新运行完整初始场景观察。

### 4.3 分阶段真机验证

第一次必须使用空容器或替代道具、低速、急停可触达，并按顺序逐阶段验证：

```bash
python run_three_stage_demo.py --observe 0 --execute --stage stopper
python run_three_stage_demo.py --observe 0 --execute --stage bottle
python run_three_stage_demo.py --observe 0 --execute --stage dropper
```

程序从当前实际 TCP 位姿重新规划，第一段通常是原地升至安全转运高度；当前位置正上方必须无障碍。每阶段和关键动作要求键入指定确认文本，其他输入会停止。停止或异常后程序不会自动松爪。

四阶段完整命令：

```bash
python run_three_stage_demo.py --observe 1 --execute --stage all
```

只有前三阶段已分别验证后才应使用完整命令。

所有现场检查已经完成、需要连续执行且不等待任何键盘输入时，显式增加 `--yes`：

```bash
python run_three_stage_demo.py --observe 0 --execute --stage all --yes
```

`--yes` 会自动通过初始观察、阶段确认、抓取检查和搅拌检查点，但不会绕过配置/几何校验、控制器错误、保护停止或急停。动作本身需要的夹爪稳定、倾倒保持和滴液等待仍会保留。

### 4.4 玻璃棒搅拌

玻璃棒不能复用初始复杂场景图像。滴管归位后需要重新观察：

```bash
python run_three_stage_demo.py --observe 0 --stage stirring --stir-observe 1
```

该命令会移动到最终观察位、采集并请求 `/v1/stirring/jobs`，但没有 `--execute` 时不会抓取。先检查终端给出的 `glass_rod/geometry_overlay.png`。

复用与当前烧杯任务严格匹配的最近结果：

```bash
python run_three_stage_demo.py --observe 0 --stage stirring --stir-observe 0
```

单独执行搅拌：

```bash
python run_three_stage_demo.py --observe 0 --stage stirring --stir-observe 1 --execute
```

必须实测玻璃棒长度、TCP 到夹持中心的变换、夹爪闭合轴、杯口安全内半径、杯底净空、插入深度和搅拌半径，并满足：

```text
stir_radius + endpoint_uncertainty + rod_radius < beaker_safe_inner_radius
```

## 5. 旧两目标滴定流程

只采集并保存输入，不访问服务器或运动：

```bash
python run_titration.py --capture-only
```

采集、上传、下载并规划：

```bash
python run_titration.py
```

复用已完成定位的静止场景：

```bash
python run_titration.py --reuse-capture captures/CAPTURE_ID
python run_titration.py --execute-saved captures/CAPTURE_ID
```

复用前必须确认物体和机器人底座没有移动。程序会按当前 TCP、当前位置和配置重建路径，不复用旧 motion plan。

重新采集并真机执行：

```bash
python run_titration.py --execute
```

执行顺序为开爪、接近胶头、夹持、提起、移动到杯口、挤压、恢复夹持和抬起；结束时仍可能夹持滴管。固定挤压次数不等于真实滴数。

已保存任务可独立查询或下载：

```bash
python client.py status JOB_ID
python client.py download JOB_ID --output-dir captures/CAPTURE_ID/server_result
```

## 6. 原始深度胶头诊断

离线读取最近一次文件完整的 RGB-D 采集，不连接硬件，也不依赖服务器解算成功：

```bash
python run_depth_bulb_grasp.py
python run_depth_bulb_grasp.py --capture-dir captures/CAPTURE_ID
```

结果写入采集目录的 `raw_depth_bulb_diagnostic/`。默认在整图搜索红/粉色胶头，也可通过 `--roi X0 Y0 X1 Y1` 指定区域；存在多个面积相近的候选时要求用 ROI 消除歧义。中心由连续有效原始深度点的轴向密度主峰附近加权得到，是可见表面估计，不是胶头完整体积中心。RGB 只用于选择橡胶像素，不调用 Qwen、SAM 或模型解算。

直接运行模块文件也使用同一个命令行入口：

```bash
python /home/dsj/vla/affordance/affordance/depth_bulb_grasp.py
```

在当前 UR 位姿重新拍摄但不移动：

```bash
python run_depth_bulb_grasp.py --live
```

移动到当前固定观察位后重新拍摄、直接用原始深度定位胶头，但不抓取：

```bash
python run_depth_bulb_grasp.py --live --observe
```

检查遮罩和坐标后，可选真实夹持测试：

```bash
python run_depth_bulb_grasp.py --live --observe --execute
```

该模式先移动到固定观察位，再重新采集、定位并接近胶头；在预抓取高度和抓取高度分别要求确认后才闭爪。追加 `--yes` 可自动通过确认点。它不会抽出、转运或自动松爪。用于标定检查时，默认 `--grasp-z-offset=0`，不会隐含下降 10 mm；需要侧面夹持补偿时必须显式传入，例如 `--grasp-z-offset -0.010`，此时最终目标已包含人为偏移。

诊断目录包含遮罩叠加图（绿色点为加权中心的投影）、实测胶头点云 `raw_depth_bulb_points_world.ply`、记录相机/世界坐标、当前手眼矩阵及哈希、TCP 偏移的 `result.json`。执行时另存 `grasp_plan.json` 与 `grasp_completed.json`。控制器实际 TCP 接近命令目标，只能说明运动到位；手眼标定、物理 TCP、深度误差及可见表面到体积中心的差异，仍需结合实际夹指落点判断。当前 TCP 配置是 150 mm，脚本不会自动改动它。

只有显式使用离线选项 `--compare-server` 才额外读取模型胶头区域做对照；默认纯深度路径不受服务器类别互换影响。启动时会检查 `hand_eye.yaml`、`T_ee_cam.json` 与 TCP 偏移是否数值一致。

### 不经过服务器的原始深度瓶塞抓取

独立入口 `run_depth_stopper_grasp.py` 会移动到固定观察位，在配置 ROI 内用白色与最近深度簇定位瓶塞顶面，并通过本地手眼矩阵得到 base 抓取点；不访问服务器、不使用重建模型。第一次只观察和规划：

```bash
python run_depth_stopper_grasp.py --capture-only
```

检查新采集目录中的 `direct_stopper_mask_overlay.png` 和 `direct_stopper_plan.json`。只有黄色 ROI 紧密包含白色瓶塞、洋红遮罩覆盖瓶塞顶面、绿色十字位于中心时，才可在空瓶/替代物条件下执行：

```bash
python run_depth_stopper_grasp.py --execute
```

人工参数位于 `config/depth_stopper_grasp.yaml`：`roi_px`、HSV 阈值、深度范围/聚类容差、可见顶面到夹持中心的 `grasp_offset_base_m`、速度和加速度。接近距离、拔起距离、夹爪位置与力复用 `config/three_stage_demo.yaml` 和 `config/config.yaml`。脚本在观察移动、下降和拔起前分别要求键入确认；成功后保持夹爪闭合并继续夹住瓶塞。

## 7. 坐标系与手眼标定

### 桌面米色橡胶塞的原始深度抓取

`run_depth_tabletop_grasp.py` 从当前视野的 RGB-D 拟合桌面，按到桌面的高度分离凸起物，再以米色/低饱和颜色辅助选择完整目标。`--color none` 可关闭颜色筛选，只使用深度；多目标时可用 `--roi X0 Y0 X1 Y1` 或 `--target-pixel U V` 指定目标。默认中心为前述轴向主峰局部加权中心，`--center mean` 可改为观测点云的算术均值，`--center median` 为中位数。

离线处理一份采集，不连接硬件：

```bash
python run_depth_tabletop_grasp.py --capture-dir captures/CAPTURE_ID
```

`--live` 从机械臂当前位置重新采集，不移动到历史观察位。`--live --execute` 才接近并闭爪，要求用 `--finger-below-tcp-mm` 填入朝下抓取姿态下夹指最低端相对 TCP 的实测向下伸出量；只有 TCP 就在最低指尖时才能填 0。程序检查估计指尖桌面间隙，并加入 3 mm 净空、控制器位置容差和桌面拟合容差，保留原工作区下限，不会自动降低目标或放宽边界。

操作者已现场明确确认指尖不会碰桌、但未提供伸出量时，可显式使用 `--operator-confirmed-table-clearance`。计划会注明现场确认且未测量，不把缺失的尺寸伪记为 0；该参数不能覆盖已有测量判定的碰桌风险、TCP 本身的桌面净空限制、工作区或 IK/关节限位检查。`--yes` 本身不会替代这个确认。

诊断产物位于采集目录的 `tabletop_stopper_diagnostic/`：`mask_overlay.png`、`points_world.ply`、`estimate.json` 与 `grasp_plan.json`。下降速度为 0.01 m/s，默认闭爪后保持在目标处，不抬起；需要抬起时显式指定 `--lift-mm`。`--yes` 自动通过各确认点。执行仍须确保夹爪空置、转移路径清楚；点云中心是可见表面统计值，完成运动不等于已独立测得抓取成功。

本项目约定：

```text
world = UR base
p_tool0 = T_ee_cam * p_opencv_camera
p_base = T_base_tool0 * T_ee_cam * p_opencv_camera
```

服务器内部若需要从 SAM3D 坐标转到 OpenCV optical，只能应用一次转换；本地收到世界坐标后不再翻转或重复乘手眼矩阵。

`server.ee_frame: flange` 在本项目中表示 UR 控制器全零 TCP 的 `tool0`，不是另一个轴向不同的 ROS flange link。配置了非零 TCP 时，本地按 `T_base_tcp * inverse(T_tool0_tcp)` 得到上传的 `T_base_tool0`。

修改 TCP、相机安装、内参、marker 尺寸或标定文件后，必须用多个静止视角和可独立测得 base 坐标的固定标志点重新验证。服务器标定文件不会由本地客户端自动上传；两端版本需要人工核对。

## 8. 安全边界

当前实现仍是开环实验系统：

- 没有完整的机械臂、夹爪、相机、线缆和环境连续碰撞模型；
- 没有可靠的夹爪接触力或抓取成功闭环；
- 没有液面、液流、重量、真实滴数或滴定终点检测；
- 倾倒量由姿态和时间近似控制，搅拌深度依赖人工测量；
- 点云拟合和工作空间检查不能替代实物尺寸与净空检查；
- 旧世界坐标只适用于物体和机器人底座都未移动的场景；
- 异常时会尽力停止机械臂并记录失败，但不会自动释放已夹物体。

真机运行时确保只有一个程序控制 UR，示教器使用低速，先逐路点验证，操作员始终能触达急停。任何服务器失败、结果缺失、坐标非有限、IK/关节限位失败、夹爪宽度异常或人工检查未通过，都应停止动作。

## 9. 测试与故障排查

### 抓取中心与烧杯接液位置

瓶塞、瓶身、胶头的点云抓取中心统一使用“沿物体轴向找密度主峰，再对峰值附近的 XYZ 点加权”。权重是 affordance 分数乘以轴向高斯权重；零分点不参与，候选点先受工作区约束。没有分数的旧结果使用等权，并在计划的 `quality_report.grasp_centers` 中注明。`grasp_center.bandwidth_m` 默认 3 mm，`bin_width_m` 默认 0.5 mm。两个独立的原始深度抓取脚本也调用此算法；原有实际夹持偏移仍按各自配置应用。玻璃棒端点服务直接返回几何抓取候选，不把端点坐标当成点云重新求中心。

烧杯接液位置不使用杯壁 affordance 的中心。程序提取完整烧杯模型的上沿外轮廓，估计杯口平面和圆心，并在该平面内生成虚拟空心开口圆盘。`beaker_opening.wall_margin_m` 和 `receiving_margin_m` 分别从估计半径扣除壁厚余量与落点余量；它们是配置假设，并非测得的玻璃壁厚。倒液默认 `bottle.pour_side_offset_m: 0`，落点在开口中央；非零偏移会限制在内缩后的开口半径内。滴液继续瞄准同一个开口中心。

离线检查，不连接机械臂、相机或服务器：

```bash
python run_three_stage_demo.py --observe 0 --stage all
```

结果目录中会生成 `local_three_stage_plan.json`、`local_beaker_opening.json` 和 `local_beaker_opening_world.ply`。PLY 是标明 `SYNTHETIC` 的接液自由空间，不是服务器测得的表面点云；服务器原始 affordance 与结果文件不会被替换。这个本地生成的开口只改变本地规划目标，服务器原始杯壁投影图仍保持原样。

上述模型估计不替代瓶盖厚度、夹指尺寸和实际 TCP 标定；密度主峰接近瓶盖顶面时，实际侧面接触位置仍受这些物理尺寸影响。

### 深度与拟合模型的彩色 PLY 对齐检查

将最新观测的完整原始深度点云与服务器拟合的三个 world 模型合并为一个 PLY：

```bash
python export_alignment_ply.py
```

默认颜色：深度灰色、广口瓶红色、烧杯蓝色、滴瓶洋红色。输出位于最新结果的
`diagnostics/depth_model_alignment_world.ply`，并额外生成更适合 VS Code 3D 预览器的同名 `.glb`。同时生成同名 JSON，记录点数、颜色、坐标系和服务器实际使用的相机世界变换。可用 `--capture-dir` 与 `--result-dir` 指定其他成对结果；两者必须属于同一次采集。

无硬件单元测试：

```bash
python -B -m unittest discover -s tests -v
```

测试使用模拟硬件和 HTTP 响应，不代表真实标定、碰撞或真机动作已经验证。

常见问题：

- `No such file or directory: /dev/serial/...`：配置的夹爪串口不存在，重新识别设备，不是 Python 错误；
- `Permission denied: /dev/ttyUSB*`：检查当前用户是否属于 `dialout` 组；
- API 超时：只表示本地停止等待，不一定取消服务器任务，先用保存的 job ID 查询；
- 结果坐标明显异常：停止执行，核对 `world/base`、`ee_frame`、TCP offset、depth scale、手眼方向和服务器标定版本；
- 计划成功但现场不安全：以现场判断为准，计划器没有完整碰撞模型。
