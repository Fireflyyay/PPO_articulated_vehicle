# PPO articulated vehicle (PPO + Motion Primitives)

该项目基于 Gymnasium 自定义环境训练一个铰接车（牵引车-挂车）智能体，核心算法为 PPO。与“直接输出连续控制”不同，本仓库默认启用 **Motion Primitives（离散宏动作）**，并结合 **全局软引导**、**action mask**、**Extrem teacher success band** 与 **自适应 primitive 扩展** 等机制提升训练稳定性与难场景表现。仓库中仍保留 **terminal takeover（末端接管规划器，RHP 快速裁剪）** 的相关实现文件，但当前默认配置下该机制处于禁用状态，不能作为现行可用功能使用。

## 依赖

- Python 3.8
- 依赖安装：
  ```bash
  conda create-n PPOMP python=3.8
  conda activate PPOMP
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

## 训练增强机制

除了 primitive 宏动作外，当前默认训练链路还依赖以下几个机制：

### 1) 全局软引导（Soft Global Guidance）

全局引导由 [src/env/global_guidance.py](src/env/global_guidance.py) 提供，默认开启。其思路是：

- reset 时先做一次粗网格 A*，生成一条全局参考路径
- step 时从参考路径中提取局部方向提示，而不是强制跟踪硬路点
- 引导特征会并入观测向量，当前默认维度为 4：方向、横向误差和提示强度等

这套机制是“建议式”而不是“硬约束式”的，主要用于减少探索早期在大场景中的无效游走。与此同时，场景采样阶段还会结合 guidance 成功性过滤掉部分明显不可学的导航样本。

### 2) Extrem teacher success band

针对 Extrem 难度，训练脚本中加入了 teacher success band 机制，用来把训练重心维持在“有挑战但仍可学习”的成功率区间附近。默认配置位于 [src/configs.py](src/configs.py#L180)：

- EXTREM_SUCCESS_BAND = (0.20, 0.60)
- EXTREM_SUCCESS_BAND_FOCUS_PROB = 0.65
- EXTREM_SUCCESS_BAND_BRIDGE_PROB = 0.75

它的目标不是一味采样最难场景，而是优先把样本分布控制在 agent 还能持续获得改进信号的 band 内，从而避免 Extrem 训练过早退化成纯失败数据。

### 3) Action mask

默认配置下 USE_ACTION_MASK = True。训练、评估和可视化在选择 primitive 前，会先通过 env.get_action_mask(obs) 过滤掉当前状态下明显不可行的宏动作，再把 mask 传给 PPO 的离散策略分布。

当前实现位于 [src/env/wrappers/macro_action_wrapper.py](src/env/wrappers/macro_action_wrapper.py#L992) 与 [src/model/agent/ppo_agent.py](src/model/agent/ppo_agent.py#L245)，核心思路是：

- 先根据 primitive 库索引做快速候选筛选
- 再按配置选择 fast_only / hybrid / full 三种模式进行不同强度的可行性检查
- 对被判定为不可行的 primitive，在策略 logits 上直接置为极小值，不参与采样

其中 action mask 的总体思路参考了 HOPE 项目：<https://github.com/jiamiya/HOPE>。更准确地说，本仓库借鉴的是“在离散路径规划动作空间中先做动作可行性筛选，再交给策略网络决策”的方法，但具体实现已经结合当前 primitive 库、索引结构和 PPO 训练流程做了本地化修改。

### 4) Primitive refinement

除上述机制外，仓库当前实际可用的末端改进能力主要来自 primitive refinement，而不是 RHP takeover。refinement 会在 primitive 计划执行前后做连续细化与 terminal polishing，以改善最终停靠质量。

## Terminal takeover（末端接管 / RHP 规划器，当前不可用）

仓库中保留了 [src/terminal_takeover_rhp.py](src/terminal_takeover_rhp.py) 与 wrapper 内的接管相关代码，但 **当前默认配置下 RHP 接管机制不可用**。直接原因是 [src/configs.py](src/configs.py#L310) 中默认设置了 TAKEOVER_ENABLE=False，因此默认训练、评估和可视化流程里不会实际触发 takeover，相关统计保持为 0 属于预期行为。

如果你只是阅读代码，可将 RHP 部分理解为保留中的实验实现，而不是当前主流程能力。

## Primitive 离线网格索引（.grid_index.npz）

离线网格索引最初用于支持 RHP 接管规划器，也可用于当前 action mask / primitive 候选筛选相关逻辑。PrimitiveLibrary 会自动尝试加载与 primitive 同名的索引文件：

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

- src/configs.py：全局配置（车辆参数/环境参数/PPO 超参/primitive、guidance、action mask、refinement 与 takeover 开关）
- src/env/：环境与车辆模型
  - env/car_parking_base.py：主环境
  - env/global_guidance.py：全局软引导（粗网格 A* + step 级方向提示）
  - env/vehicle.py：铰接车运动学与状态
  - env/wrappers/macro_action_wrapper.py：宏动作（primitive）包装、action mask 与保留中的 takeover 分支
- src/model/：PPO agent 与网络
  - model/agent/ppo_agent.py：PPO 算法（支持离散 primitive policy 的 action mask）
  - model/agent/parking_agent.py：宏动作队列执行器（PrimitivePlanner）
- src/primitives/：primitive 库与索引
  - primitives/library.py：加载 .npz primitive 库（自动尝试加载 .grid_index.npz）
  - primitives/primitive_index.py：网格索引结构与加载
- src/primitives/primitive_refinement.py：primitive plan 的连续细化/终端 polishing
- src/train/train_ppo.py：训练入口（包含 teacher success band、日志、保存与可选自适应扩展）
- src/evaluation/visualize_path.py：可视化与保存轨迹图到 src/img/
- src/terminal_takeover_rhp.py：RHP 接管规划器实现文件，当前默认流程不启用
- scripts/build_primitive_grid_index.py：离线构建 .grid_index.npz

## 测试

仓库包含若干针对 primitive / takeover 的单测，可在根目录运行：

```bash
pytest -q
```