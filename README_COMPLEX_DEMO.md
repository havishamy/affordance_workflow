# 四阶段复杂实验（不改变原 workflow）

新入口为 `run_complex_demo.py`。原 `run_titration.py`、`affordance/workflow.py` 和旧双目标 API 均不修改。

## 真实能力边界

服务器只提供：

- 物体三维模型/点云；
- 模型尺度和位姿；
- 广口瓶塞子、瓶身、旧实验烧杯口、滴管胶头、玻璃棒两端等已有 affordance 区域。

服务器不提供瓶口轴向、烧杯口法向/半径、液面、滴管尖端、抓取姿态、滴数或滴定终点。烧杯接液中心来自旧实验已有的烧杯口 affordance 区域。

本地从 affordance 区域点集计算中位数中心，并结合针对固定 canonical 模型人工标定的几何先验，推导动作需要的近似锚点。详细服务端契约见 `SERVER_COMPLEX_DEMO_IMPLEMENTATION_SPEC.md`。

## 本地推导，不是服务器预测

```text
塞子抓取点 = median(stopper region world points)
瓶身抓取点 = median(body_grasp region world points)
瓶轴 = server model pose/scale × local canonical bottle axis
瓶口近似点 = 塞子中心 + 本地标定的有符号偏移

烧杯口中心 = median(server beaker mouth affordance world points)
烧杯轴/安全内半径 = local measured cell parameters

滴管胶头点 = median(bulb region world points)
滴管尖端 = server model pose/scale × local canonical tip anchor

玻璃棒两端 = median(end_a/end_b region world points)
抓取端 = 本地配置选择，默认 world Z 较高的一端
```

没有液面检测。搅拌深度从烧杯口 affordance 中心沿本地配置轴向下固定距离 `target_depth_below_mouth_m`，属于开环参数。

## 四个动作阶段

1. `open_bottle`：抓塞子 affordance 中心，沿本地推导的 canonical 瓶轴拔出，放到人工标定停车位。
2. `pour`：抓瓶身 affordance 中心，将本地推导的近似瓶口保持在烧杯口 affordance 中心附近，分步倾斜并把瓶子放回。
3. `dropper`：抓胶头 affordance 中心，用 canonical 滴管尖端先验对准烧杯口 affordance 中心，按配置执行夹爪挤压，再放回。
4. `stir`：抓玻璃棒一端，把另一端移动到烧杯口 affordance 中心下方，执行配置圆轨迹并放回。

这些动作都依赖固定对象、稳定 canonical 模型和人工几何标定，不是通用任意物体操作。

## 服务端最小输出

新端点为 `/v1/complex-demo/jobs`。结果只包含：

```json
{
  "frame": "world",
  "units": "meters",
  "model_transform_convention": "p_world = T_world_object @ [scale * p_object, 1]",
  "objects": {
    "wide_mouth_bottle": {
      "model_id": "wide_mouth_bottle_v1",
      "T_world_object": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "scale": [1,1,1],
      "model_points_world_artifact": "models/wide_mouth_bottle_world.ply",
      "affordance_regions": {
        "stopper": {"points_world": [[0,0,0]]},
        "body_grasp": {"points_world": [[0,0,0]]}
      }
    },
    "beaker": {
      "model_id": "beaker_v1",
      "T_world_object": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "scale": [1,1,1],
      "model_points_world_artifact": "models/beaker_world.ply",
      "affordance_regions": {"mouth": {"points_world": [[0,0,0]]}}
    },
    "dropper": {
      "model_id": "dropper_v1",
      "T_world_object": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "scale": [1,1,1],
      "model_points_world_artifact": "models/dropper_world.ply",
      "affordance_regions": {"bulb": {"points_world": [[0,0,0]]}}
    },
    "glass_rod": {
      "model_id": "glass_rod_v1",
      "T_world_object": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "scale": [1,1,1],
      "model_points_world_artifact": "models/glass_rod_world.ply",
      "affordance_regions": {
        "end_a": {"points_world": [[0,0,0]]},
        "end_b": {"points_world": [[0,0,0]]}
      }
    }
  }
}
```

实际区域必须包含足够多的点；示例中的单点只是省略写法。

## 必须先标定的本地先验

`config/complex_demo.yaml` 中以下内容目前都是占位值：

- 四个精确 `model_id`；
- canonical 瓶轴；
- 塞子 affordance 中心到真实瓶口的有符号偏移；
- 烧杯 world 轴方向和实测安全内半径；
- canonical 滴管尖端锚点；
- 玻璃棒抓取端选择；
- 瓶塞停车位；
- 四种抓取姿态和夹爪位置；
- 倾倒角度/时间、滴管挤压参数、搅拌下探深度。

如果服务器重建的 canonical 坐标每次不稳定，这些先验无法复用，当前复杂实验不能安全执行。必须先让服务器明确模型 pose/scale 的定义与稳定性。

## 离线规划

示例 JSON 只用于格式和几何代码测试：

```bash
conda activate ur_data
cd /home/dsj/vla/affordance
python run_complex_demo.py --scene-json config/complex_scene_example.json
```

该命令连接 UR 读取当前 TCP 并设置 TCP/负载，但不会发送运动。

选择部分阶段：

```bash
python run_complex_demo.py \
  --scene-json REAL_RESULT.json \
  --stages open_bottle,pour,dropper
```

服务器接口完成后，不加 `--scene-json` 即可采集、上传、下载和规划。

## 真机执行闸门

当前配置：

```yaml
physical_accuracy_validated: false
```

因此 `--execute` 会在连接硬件前被拒绝。只有服务器 model ID/pose/scale 契约、所有 canonical 先验、物体偏移、连续路径和替代道具实验全部验证后才能设为 true。

执行时每阶段分别要求：

```text
EXECUTE OPEN_BOTTLE
EXECUTE POUR
EXECUTE DROPPER
EXECUTE STIR
```

## 明确的开环限制

- 瓶口是塞子区域加人工偏移，不是服务器检测；
- 烧杯口中心来自服务器旧 pipeline 的 mouth affordance 区域；法向和半径仍不是服务器输出；
- 滴管尖端来自 canonical 模型锚点，不是服务器关键点；
- 搅拌深度相对杯口固定，不知道真实液面；
- 挤压次数不等于视觉确认的真实滴数；
- 倾倒角度/时间不等于定量体积；
- 没有完整碰撞规划、抓取成功检测或阶段间重新定位；
- 一次初始拍摄用于全部阶段，放回误差会累积。
