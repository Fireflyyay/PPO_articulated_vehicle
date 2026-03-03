# PPO articulated vehicle (PPO + Motion Primitives)

该项目基于 Gymnasium 自定义环境训练一个铰接车（牵引车-挂车）智能体，核心算法为 PPO。与“直接输出连续控制”不同，本仓库默认启用 **Motion Primitives（离散宏动作）**，并带有 **terminal takeover（末端接管规划器，RHP 快速裁剪）** 与 **自适应 primitive 扩展（可选/默认开启，见配置）**。

## 依赖

- Python 3.8+
- 依赖安装：
  ```bash
  pip install -r requirements.txt
  ```

依赖列表以 requirements.txt 为准（torch / gymnasium / shapely / pygame / matplotlib / tensorboard / einops / tqdm 等）。

## 快速开始

### 1) 训练

训练入口为 src/train/train_ppo.py。脚本会自动将 src 加入 sys.path，因此可以在仓库根目录直接运行：

```bash
python src/train/train_ppo.py
```

常用参数（以代码实际参数为准）：

```bash
# 训练指定回合数
python src/train/train_ppo.py --train_episode 100000

# （可选）每次评估使用的回合数（训练脚本内部会用到）
python src/train/train_ppo.py --eval_episode 100

# 从 checkpoint 恢复（params_only=True 保存格式）
python src/train/train_ppo.py --agent_ckpt /path/to/PPO_best.pt

# 注意：这里 argparse 使用 type=bool，需显式传 True/False
python src/train/train_ppo.py --verbose True --visualize False
```

日志与模型保存：

- TensorBoard 日志目录：src/log/exp/ppo_YYYYMMDD_HHMMSS/
- 最优模型：当三个场景（Normal/Complex/Extrem）最近 100 回合成功率“均不低于历史最佳”时，保存为 src/log/exp/ppo_*/PPO_best.pt
- 周期性快照：每 2000 episode 保存一次 PPO2_<episode>.pt

训练脚本启动时会打印可用的 TensorBoard 命令（python -m tensorboard --logdir ...）。

渲染/显示说明：src/configs.py 默认设置了 SDL_VIDEODRIVER=dummy 以支持无显示环境运行。如果你希望在本机弹出 pygame 窗口进行交互式渲染，需要自行调整该环境变量设置。

### 2) 可视化轨迹（评估）

可视化脚本为 src/evaluation/visualize_path.py，会把每个 episode 的轨迹图保存到 src/img/。

```bash
python src/evaluation/visualize_path.py --episodes 10
python src/evaluation/visualize_path.py --episodes 5 --level Complex
```

注意：该脚本默认只会在 src/ckpt/ 下查找 PPO_best.pt（找不到会退出）。如果你想可视化刚训练出来的模型，请将训练输出的 PPO_best.pt 复制/软链接到 src/ckpt/PPO_best.pt，或自行修改脚本中的 _find_checkpoint 逻辑。

## Motion Primitives（离散宏动作）

默认配置在 src/configs.py：

- USE_MOTION_PRIMITIVES = True
- PRIMITIVE_LIBRARY_PATH = "../data/primitives_articulated_H4_S11.npz"（相对 src/ 的路径）

启用后：

- 环境会被 src/env/wrappers/macro_action_wrapper.py 的 MacroActionWrapper 包装
- PPO actor 输出维度会变为 env.action_space.n（即 primitive 个数），动作是“primitive id”而不是连续 (steer, speed)
- wrapper 会把 primitive 的物理动作序列转换/归一化为 env.step 所需的 [-1, 1] 范围后逐步执行

仓库已提供 data/ 下的 primitive 库文件。

## Terminal takeover（末端接管 / RHP 规划器）

当启用 Motion Primitives 时，MacroActionWrapper 可能在“接近目标”或“困难状态”自动触发接管，使用 src/terminal_takeover_rhp.py 的 RecedingHorizonTakeoverPlanner 进行在线短视距规划。

触发逻辑（简述，具体以代码为准）：

- 距离触发：dist <= TAKEOVER_DIST_BASE（带 hysteresis 与可选速度/障碍密度增益）
- 困难触发：相对航向误差、铰接角偏差、最小 lidar 距离等超过阈值

关键开关位于 src/configs.py：

- TAKEOVER_USE_RHP：是否启用 RHP 接管规划器
- TAKEOVER_DIST_BASE / TAKEOVER_DIST_HYSTERESIS / TAKEOVER_EARLY_*：触发阈值
- OCCUPANCY_INFLATION_RADIUS / TAKEOVER_SCORE_WEIGHTS：碰撞裁剪与打分权重

## Primitive 离线网格索引（.grid_index.npz）

RHP 接管规划器支持读取“离线网格索引”以加速在线碰撞裁剪。PrimitiveLibrary 会自动尝试加载与 primitive 同名的索引文件：

- <library>.grid_index.npz（例如 data/primitives_articulated_H4_S11.grid_index.npz）

仓库已带一个默认索引文件；如需为自定义 primitive 重新生成索引，可运行：

```bash
python scripts/build_primitive_grid_index.py \
  --library data/primitives_articulated_H4_S11.npz \
  --grid_resolution 0.3
```

输出默认写回到与 library 同目录同名的 .grid_index.npz，并会在运行时被自动加载。

## 自适应 primitive 扩展（Adaptive Primitive Expansion）

src/configs.py 中 USE_ADAPTIVE_PRIMITIVE_EXPANSION 默认开启，会在训练过程中自动：收集 rollouts → 挖掘候选片段 → 去重/剪枝/可行性检查 → 扩展离散动作空间（actor 输出维度）→ 必要时回滚。

如果你只想跑一个稳定的 baseline（固定 primitive 集），可将以下开关关闭：

- USE_ADAPTIVE_PRIMITIVE_EXPANSION = False
- （可选）USE_DISCOVERED_PRIMITIVE_SHAPING = False

## 代码结构

- src/configs.py：全局配置（车辆参数/环境参数/PPO 超参/primitive 与 takeover 设置）
- src/env/：环境与车辆模型
  - env/car_parking_base.py：主环境
  - env/vehicle.py：铰接车运动学与状态
  - env/wrappers/macro_action_wrapper.py：宏动作（primitive）包装 + 接管逻辑
- src/model/：PPO agent 与网络
  - model/agent/ppo_agent.py：PPO 算法
  - model/agent/parking_agent.py：宏动作队列执行器（PrimitivePlanner）
- src/primitives/：primitive 库与索引
  - primitives/library.py：加载 .npz primitive 库（自动尝试加载 .grid_index.npz）
  - primitives/primitive_index.py：网格索引结构与加载
- src/train/train_ppo.py：训练入口（日志/保存/可选自适应扩展）
- src/evaluation/visualize_path.py：可视化与保存轨迹图到 src/img/
- scripts/build_primitive_grid_index.py：离线构建 .grid_index.npz

## 测试

仓库包含若干针对 primitive / takeover 的单测，可在根目录运行：

```bash
pytest -q
```