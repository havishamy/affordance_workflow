# 四阶段真机实验：拔塞、倾倒、滴液、玻璃棒搅拌

为兼容已有现场命令，入口和配置文件仍保留 `run_three_stage_demo.py`、`three_stage_demo.yaml` 名称；流程现已扩展为四阶段。旧 `run_titration.py` 和旧 `affordance/workflow.py` 不受影响。

采集时的 TCP offset 为 `[0,0,0.140,0,0,0]`。现场当前 offset 可以不同，程序不再要求两者相等：服务器结果是 UR base 中的物体坐标，执行时会使用**当前实际 TCP 位姿**重新规划。但这要求当前设置的 TCP 确实对应实际夹爪夹持中心；若只是修改数字而夹持中心未变，抓取会出现对应偏差。离线路径的首段仍用历史拍照位姿作预览，执行前另存的 `plan_<stage>.json` 才是从当前位姿生成的动作计划。

## 结果核查与本地几何

输入是 `config/three_stage_demo.yaml` 指向的已解压结果。当前 job 为 `df530c2b-86be-4cc5-82b1-646908ab7886`，三个模型的轮廓 IoU 约 0.788/0.800/0.819。瓶塞、瓶身、烧杯口、胶头 affordance 的二维峰值分别落在各自操作区域；瓶塞与瓶身的 top-region IoU 为 0。

本地不会使用服务器不稳定的 SAM3D canonical 坐标系：

| 量 | 当前来源 | 约值，UR base / m |
|---|---|---|
| 瓶塞夹持 XY | 重建瓶体顶部轴线；校验与瓶塞区域相近 | `[0.6572, -0.1280]` |
| 瓶塞夹持 Z | 瓶塞 affordance 区域中位数 | `0.1014` |
| 瓶身夹持 | 同瓶体轴线 XY + body_grasp 区域高度 | `[0.6572, -0.1280, 0.0592]` |
| 近似瓶口 | 瓶塞夹持点沿瓶轴减去配置偏移 | `[0.6572, -0.1280, 0.0914]` |
| 烧杯接液中心 | 对服务器 mouth affordance 的三维部分圆弧拟合圆心 | `[0.6509, -0.0224, 0.0854]` |
| 胶头夹持 | bulb affordance 区域中位数 | `[0.5505, -0.1587, 0.1837]` |

烧杯 mouth 区域在本次只覆盖杯口一侧。直接取区域中位数会偏向杯沿约 27 mm，故本流程拟合 XY 圆心；圆半径约 29.7 mm，残差约 0.37 mm，圆弧覆盖约 1.94 rad。杯口 Z 取该区域的第 98 百分位。这个数学拟合良好，但不能替代实物绝对坐标测量。

## 抓瓶策略

选择“从上方竖直接近，夹爪指尖在瓶身中段横向闭合”，不是从侧面水平靠近，也不是只捏瓶口顶部。理由：瓶体点云宽约 58 mm、高约 95 mm，抓取中心高度约 59 mm；从侧面水平接近时夹爪/腕部更容易与桌面和相邻物体干涉。从上方接近后横向夹持瓶身，可先竖直提离桌面，再绕水平轴倾斜。此判断只覆盖瓶体几何；真实夹爪开口、指尖长度、颈部净空必须现场检查。

倾倒采用“高处预倾斜 → 保持瓶口世界坐标不变 → 低速下降 → 保持 → 抬高 → 恢复直立”的顺序。每个倾斜路点用 TCP 平移补偿瓶口相对夹持中心的旋转。瓶口目标先放在烧杯口偏向广口瓶的一侧，配置 `pour_side_offset_m` 控制该侧向偏移。瓶体点云代理会提示桌面高度和杯口圆柱侵入风险，但这是估计警告，不是完整机械臂、夹爪、液体或连续轨迹碰撞模型。

## 运行方式

`config/three_stage_demo.yaml` 的 `observation.enabled` 只能是 `0` 或 `1`，命令行 `--observe 0|1` 可临时覆盖。

| 模式 | 行为 |
|---|---|
| `0`（默认） | 优先读取最近一次模式 1 成功解算并通过基础格式/几何检查的本地结果；若尚无新结果则读取 YAML 中的旧 `capture_dir/result_dir`。不连接相机、不访问服务器；仅加 `--execute` 才执行阶段动作。 |
| `1` | 先检查服务器/Token，确认后移动 UR 到参考观察位姿，采集 RGB-D、上传解算、下载并验证完整结果，再规划；即使未加 `--execute`，观测阶段仍会移动机械臂。加 `--execute` 才继续执行阶段动作。 |

模式 0：

```bash
conda activate ur_data
cd /home/dsj/vla/affordance
python run_three_stage_demo.py --observe 0
```

模式 1：先保持复杂实验服务器和 SSH 隧道运行，并设置 `LAG_API_TOKEN`；确保瓶塞、瓶身、烧杯口、胶头都在新画面中，且机器人上方及到观察点的转运路线无障碍。

```bash
export LAG_API_TOKEN='与服务器一致的密钥'
python run_three_stage_demo.py --observe 1
```

这条命令**会移动到观察位姿**，但只完成采集、服务器解算和规划，不执行拔塞/倾倒/滴液。完整长程实验使用：

```bash
python run_three_stage_demo.py --observe 1 --execute --stage all
```

观察位姿从 `observation.reference_capture_info` 指向的 `20260921_171530_302852/capture_info.json` 读取。参考数据含拍照时 TCP 位姿及当时 `+Z 0.140 m` offset；若当前 offset 不同，程序会按同一**法兰盘/tool0 位姿**换算新的目标 TCP，避免相机位置因 TCP 数值变化而偏移。此换算只用于复现观察位姿，不会把服务器返回的 UR base 物体点再变换一次。

模式 1 的新数据保存到 `captures/<新时间>/`，服务器结果保存到其下的 `complex_server_result_with_beaker/`，下载 `scope=all` 并安全解压。四个输入文件、`capture_info.json`、观察移动路径及失败记录都会保留。服务器解算或基础数据完整性检查失败时，不进入阶段动作；感知质量低的指标会列为人工复核警告。

成功后会原子更新 `observation.latest_result_manifest` 指向刚通过基础检查的结果；下一次模式 0 自动读取这个本地指针，绝不访问服务器。设置 `observation.reuse_latest_result: false` 可强制模式 0 使用 YAML 中手动指定的旧结果。模式 1 请求仍要求瓶塞、瓶身、烧杯口、胶头四个 affordance；若瓶塞已被拔走，再拍照不能按“完整初始场景”正常解算，应使用模式 0 或准备适合新场景的服务端任务。

模式 0 生成：

```text
captures/20260921_171530_302852/complex_server_result_with_beaker/
  complex_df530c2b-86be-4cc5-82b1-646908ab7886_results/
    local_three_stage_plan.json
```

四个阶段的顺序：

1. `stopper`：打开夹爪、抬到安全高度、从上方接近瓶塞、夹持、竖直拔出、移到实测停车位并放下。
2. `bottle`：从上方接近并横向夹住瓶身、提起、到烧杯侧上方、分步倾斜并降低瓶口、等待倾倒、抬高恢复、放回原位置。
3. `dropper`：抓胶头、竖直抽出滴管、移到烧杯接液中心上方、按配置挤压、回到原瓶并松爪。
4. `stirring`：滴管归位后重新移动到观察位，调用 `/v1/stirring/jobs` 定位平放玻璃棒两端；人工核对叠加图后，从两个对称候选中进行工作空间和 UR 控制器 IK/关节限位筛选，从上方抓住一端，在安全高度将自由端调整为竖直向下，移动到原 complex-demo 烧杯口上方，浅插入并执行低速圆周搅拌，最后退出并放回原位。

## 第四阶段的独立感知

玻璃棒不能复用初始 complex-demo 图像。完整 `--stage all` 会在前三阶段和滴管归位后，自动重新拍摄并请求 CPU 路由 `/v1/stirring/jobs`。该接口只提供玻璃棒端点、轴线和两个抓取候选；本地不会把它当成可直接执行的六自由度抓取姿态。

首次只做玻璃棒观察和结果下载：

```bash
export LAG_API_TOKEN='与服务器一致的密钥'
python run_three_stage_demo.py --observe 0 --stage stirring --stir-observe 1
```

这条命令会移动到最终观察位，但不会抓取。`--stir-observe 1` 本身作为观察移动授权，不再额外要求键盘输入；下载后终端会打印 `glass_rod/geometry_overlay.png` 路径供核查。质量阈值、结果 schema、单位、坐标系、端点一致性、手眼组合矩阵和全部声明的调试图仍为硬校验；失败时不会使用旧玻璃棒结果继续。

复用最近一次与当前烧杯 complex-demo job 严格匹配的玻璃棒结果：

```bash
python run_three_stage_demo.py --observe 0 --stage stirring --stir-observe 0
```

完整四阶段命令保持为：

```bash
python run_three_stage_demo.py --observe 1 --execute --stage all
```

第四阶段分开执行：

```bash
python run_three_stage_demo.py --observe 0 --stage stirring --stir-observe 1 --execute
```

不要对单独的 `stirring` 阶段使用 `--observe 1`：瓶塞已经移走后，完整 complex-demo 初始场景不再成立。

## 搅拌参数闸门

`config/three_stage_demo.yaml` 中 `stirring_motion.validated` 默认是 `false`，所有必须实测的长度保持 `null`。这会阻止第四阶段真机动作，也会在完整 `--stage all --execute` 的前三阶段动作开始前停止。必须现场测量并填写：

- `pregrasp_height_m`：桌面上方预抓取高度；
- `lift_height_m`：长棒调姿时的提升距离；
- `preinsert_clearance_m`：自由端在杯口上方的净空；
- `target_depth_below_mouth_m`：自由端低于杯口的搅拌深度；
- `stir_radius_m`、`stir_speed_mps`、圈数和每圈采样点；
- `beaker_safe_inner_radius_m`：考虑杯壁厚度后的真实安全内半径；
- `bottom_clearance_m`：自由端距离真实杯底的安全间距（服务器不定位杯底，仍需人工确认）；
- `rod_grasp_center_tool_m`：从当前 TCP 原点到被夹住玻璃棒中心线的实测三维向量，即本地 `T_tool_rod` 的平移部分；
- `rod_grasp_position`、`rod_grasp_force` 和夹爪闭合轴约定。

必须满足：

```text
stir_radius + endpoint_uncertainty + rod_radius < beaker_safe_inner_radius
```

代码会逐路点检查 Cartesian 工作空间，并在真机连接后让 UR 控制器检查 IK 和关节安全限位；仓库仍没有完整的机械臂、相机、线缆和环境碰撞模型。第四阶段只保留两个人工键入点：TCP 已下降到玻璃棒最低抓取点、夹爪尚未闭合时；玻璃棒自由端已经对准烧杯口、尚未插入时。观察移动和服务器几何叠加图只打印提示，不再阻塞。首轮必须使用空烧杯和低速验证。

真机模式只能在现场参数验证后使用：

```bash
python run_three_stage_demo.py --execute --stage stopper
python run_three_stage_demo.py --execute --stage bottle
python run_three_stage_demo.py --execute --stage dropper
```

也可以 `--execute --stage all`，但第一次应逐阶段验证。`--stage stopper` 现在只规划并预检 stopper，不会被倾倒侧偏等 bottle 参数挡住。每阶段开始和关键动作前都需要键入 `EXECUTE <步骤名>`；程序不提供跳过确认的参数。仅输入其他字符会停止，并保留当时的夹持状态，不会擅自松爪。

每次真机阶段都从机器人的**当前实际 TCP 位姿**重新生成路径；不要求回到拍照位姿或上一阶段的结束位姿，也不会自动重置机器人。第一段运动是保持当前姿态原地升到安全转运高度，因此操作员必须确认当前位置上方的升降通道无障碍。独立执行 `bottle` 或 `dropper` 时，人工确认前序物理状态正确；旧的 `completed.json` 不能证明瓶塞或瓶子真的被抓住。

## 真机前必须完成的现场测量

`physical_execution_validated` 是真机闸门；为 `false` 时，`--execute` 在连接机器人前就会拒绝。这个现场配置可由操作员调整，运行前必须核对当前实际值；为 `true` 不代表下面的检查已由程序自动完成：

1. **静态场景**：物体和机器人底座从 2026-09-21 17:15 采集后都没有移动。当前 TCP 可与拍照位姿不同，但从当前位置原地升高和到目标的横向路径必须人工检查。如场景变化，必须重新采集/解算并更新独立配置，而不是硬用旧点。
2. **瓶塞抓取**：确认所配置下降方向、夹爪指尖长度、橡胶塞直径、抓取开度和拔出行程。夹爪位置以当前配置为准，完成过一次动作不等于抓取成功。
3. **瓶塞停车位**：必须放置实体托盘并手动示教当前配置中的 TCP 放置中心、高度及避障路径。
4. **广口瓶夹持**：实测瓶身直径约 55–58 mm，确认 PGE 夹爪最大开口与夹持力。夹爪位置以当前配置为准，先用空瓶和替代液体测试。
5. **瓶口偏移与倾倒**：瓶口从瓶塞区域中心下移 10 mm 是估计值，必须用实物测量。倾角、瓶口净空、侧偏和等待时间以当前配置为准；先慢速、少量替代液体验证。
6. **滴管尖端长度**：`tip_below_bulb_m=0.12` 是未测量值。真实 TCP 滴液高度按 `杯口高度 + 实测胶头夹持中心至尖端长度 + 尖端净空` 计算，避免尖端撞杯底/杯沿。夹持 `50`、挤压 `0` 沿用旧实验语义，但仍需现场确认新滴管尺寸。
   `dropper.grasp_offset_z_m` 只改变胶头抓取、抽出和归位高度：负值向下。当前设置 `-0.010 m`，相对服务器胶头 affordance 中心向下 10 mm；代码同时用 `tip_below_bulb_m + grasp_offset_z_m` 计算新的夹持中心到滴管尖端距离，使滴液尖端净空不因抓取点改变而暗中变化。它不能弥补错误的 TCP 标定或夹爪开度；下移后必须检查指尖不会碰玻璃瓶口。
7. **工作空间及碰撞**：点级/模型代理检查不能识别 UR 手臂本体、相机、夹爪、瓶塞停车位、桌面边缘和软管。示教器低速单步观察，急停可触达。

`quality` 段中 IoU、区域 RMS、杯口拟合残差、半径范围、瓶塞/瓶身间距等是**参考阈值**，超过只在日志和计划中告警；瓶口侧偏距拟合杯沿 3 mm、点云模型靠近杯沿/桌面，以及“模型最高点 + 80 mm”的转运高度也是告警。它们不是实物尺寸测量。仍会硬停止的包括缺少目标、非有限/不合法位姿、无法构造杯口或瓶轴、TCP 路点越过配置的工作空间、UR 保护停止和 `moveL` 失败。告警出现时须在现场人工检查，不代表可以忽略物理碰撞。

每次真机执行前，程序在 `local_execution_runs/<时间>/` 写入每阶段计划和逐步事件记录；异常时写 `failure.json` 并调用 `stopL`。它不会在异常后自动放开已夹持的物体。

## 仍然开环的部分

- 没有夹爪接触力或抓取成功闭环；
- 没有液流/重量反馈，倾倒量由角度与时间控制；
- 没有真实液滴检测，夹爪挤压次数不等于准确滴数；
- 旧场景坐标只适用于同一静止布置；
- 没有完整碰撞规划；
- 不能保证第一次真机运行就成功。必须现场标定、低速分阶段验证。
