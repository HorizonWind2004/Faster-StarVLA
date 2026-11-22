# Faster-StarVLA

[English](README.md) | [简体中文](README.zh-CN.md)

**同样的 GPU 上，starVLA 的训练步快 2.00×。由 AI agent 按 [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent) 的方法优化。**

![Faster-StarVLA：2 张 GPU 上训练步加速 2.00x](docs/faster/figures/benchmark.svg)

| 实现 | 每个训练步耗时 (ms) ↓ | 加速比 ↑ | 验证方式 |
|---|---:|---:|---|
| starVLA `99764f5` | 905.4 | 1.00× | |
| Faster-StarVLA，逐位一致的开关 | 521.9 | 1.74× | 记录的每个张量逐位相同 |
| **Faster-StarVLA，全部开关** | **453.3** | **2.00×** | 落在基线自身的种子间波动之内 |

在 **2 × NVIDIA B300** 上，用 starVLA 的 [LIBERO 配方](examples/simBenchmarks/LIBERO/README.md)测得：QwenOFT 框架、
Qwen3-VL-4B 底座、四个 LIBERO 套件合训一个策略、每卡 batch 16、DeepSpeed ZeRO-2、bf16、全部参数可训练（45.0 亿）。
参考 → 候选 → 参考，各在全新进程里运行，编译缓存为空；预热 20 步后计 2 个窗口、每个 100 步（加速比
1.9975×）。“逐位一致的开关”一行是候选的单次运行，对比优化前冻结的参考计时（906.9 ms）。全部开关打开时，
每卡显存峰值从 66.5 GiB 降到 59.0 GiB。

## 快速开始

环境、权重和数据按[上游说明](README.upstream.md)准备，然后在训练启动脚本里加一行：

```bash
source scripts/configs/faster.env            # 全部开关：2.00x，非逐位一致（含编译层）
# source scripts/configs/faster_bitwise.env  # 只开逐位一致的开关：1.74x

bash examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh
```

每个开关都是环境变量，不设置就是关：一个都不设时，训练走的就是上游代码路径。这些开关针对
**QwenOFT 框架 + Qwen3-VL 底座**；两个 DeepSpeed 开关对任何用 ZeRO-2 的 starVLA 训练都适用。

## 优化过程

![优化历史：1.00x 到 2.07x](docs/faster/figures/history.svg)

图中每个点是一次计时运行，对比冻结的参考计时（906.9 ms）；上面的标题数字来自最终的三进程检查。完整记录
（包括没有效果的尝试）见 [docs/faster/WORK_LOG.md](docs/faster/WORK_LOG.md)。

## 改了什么

每卡 batch 16 时，GPU 只有 53% 的时间在工作。训练步并不缺算力：它在等主机、在一个一个地发起上万个小算子、在来回
拷贝梯度。

- **输入处理挪到数据加载进程**（`STARVLA_PREP_IN_WORKERS`）。提示词拼接、Qwen3-VL processor 和旋转位置索引
  （每步约 0.2 秒单线程 CPU）原来在训练进程里做，GPU 只能等着。数据加载进程最先拿到样本，也有空闲，现在由它们跑
  同一段代码，直接交出张量。在训练进程里开后台线程没有用：它和受发起速度限制的前向抢同一把 GIL。
- **视觉塔**（`STARVLA_BATCHED_VISION_ATTN`、`STARVLA_VISION_POS_CACHE`）。用 SDPA 时，transformers 的 Qwen3-VL
  视觉注意力每层都要和主机同步一次，并且逐张图片各算一次注意力：每步 24 层 × 32 张图。训练里所有图片大小相同，
  所以每层合成一次批量调用即可。位置表只取决于网格大小，只建一次，不再每步重建。
- **去掉没用的计算**（`STARVLA_SKIP_LM_HEAD`、`STARVLA_FAST_ATTN_MASK`）。QwenOFT 只读最后一层隐状态，原来每个
  位置的词表 logits 都算了再丢掉。注意力掩码原来经过四层嵌套的 `torch.vmap` 构造（150 ms 主机时间），两次广播
  运算就能得到同一个掩码。
- **DeepSpeed ZeRO-2**（`STARVLA_DS_FUSED_GRAD_NORM`、`STARVLA_DS_PLAIN_ALLREDUCE`）。梯度裁剪的范数原来先把每个
  梯度转成 fp64（每步写入再读回 18 GB）；`vector_norm(g, dtype=float64)` 不做这份拷贝，结果逐位相同。梯度桶改为
  原地 all-reduce，不再先切片、拼接、归约、再拷回。
- **编译层**（`STARVLA_COMPILE_LM`、`STARVLA_COMPILE_VISION`）。36 层解码器一次前向加反向约有 1 万个 GPU 算子，
  三分之一的时间在矩阵乘，其余花在逐个发起小的逐元素算子上。每一层用 Inductor 编译（只用于训练调用，评估仍走
  eager）。编译之后，Python 垃圾回收会在个别训练步上停顿 1–2.5 秒去扫描编译器留下的对象；`gc.freeze()` 消除了
  这些停顿。

每个开关的实测效果见 [docs/faster/measurement.md](docs/faster/measurement.md#switches)。

## 验证

所有结论都来自优化开始之前就构建并冻结的评判程序（[docs/faster/judge/](docs/faster/judge/)，
[判定方式](docs/faster/measurement.md#the-judge)）。

**逐位一致的开关。** 打开七个非编译开关，在确定性模式下跑两个 2 步训练用例（其中一个在两步之间插入配方里的周期性
评估），两张卡都记录：23,336 个张量（loss，以及全部 729 个参数的 fp32 梯度、更新量和两个 Adam 状态）与上游
逐位相同；学习率、随机数状态、喂给 VLM 的输入摘要也都相同。

**全部开关。** 编译层会改变浮点舍入，所以完整开关集用统计方式比较：确定性模式下、与上游同一个种子，跑一条 60 步的
训练轨迹；每个端点与上游的差必须在上游自身种子间标准差（6 个种子，优化前冻结）的 3σ 之内。

| 端点 | 差值 / 上游种子 σ | 在 3σ 内 |
|---|---:|:---:|
| Training loss, steps 20-40 | +0.87 | ✓ |
| Training loss, steps 40-60 | +0.95 | ✓ |
| Held-out probe loss, steps 50-60 | +0.12 | ✓ |
| Action-head update size, steps 10-60 | +0.29 | ✓ |
| VLM update size, steps 10-60 | -0.40 | ✓ |
| Action-head update size, top elements, steps 1-6 (tracks the learning rate) | +0.22 | ✓ |
| VLM update size, top elements, steps 1-6 (tracks the learning rate) | -0.31 | ✓ |

把所有学习率乘以 1.02 的对照运行会被这项检查拒绝（更新量端点偏离 +16σ）；让 VLM 收不到梯度的版本、改了裁剪阈值的
版本同样被拒绝。

**LIBERO 评测。** 用全部开关完整训练一次（30,000 步，种子 42，2 × B300，全局 batch 32），按 starVLA 的 LIBERO 协议
评测（每个套件 500 个 episode），并在同一条评测管线里评测官方发布的 checkpoint。官方 checkpoint 由 starVLA 作者在
8 张 GPU 上训练（全局 batch 128，取第 50,000 步）；这里只是检查加速后的代码能训出可用的策略，不是同种子的对照。这 30,000 步
实际用时 3 小时 51 分钟（含配方里的周期性评估和三次 checkpoint 保存，平均每步 462 ms）；按冻结的上游单步耗时估算，
只算训练步就要 7.6 小时（没有实际跑）。

| Success rate (%) | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|
| starVLA Qwen3-VL-OFT, published | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| Released checkpoint (`StarVLA/Qwen3-VL-OFT-LIBERO-4in1`, step 50,000), our evaluation | 98.6 | 100.0 | 99.4 | 96.6 | 98.65 |
| **Faster-StarVLA run** (all switches, 30,000 steps on 2 GPUs) | 98.8 | 98.6 | 98.4 | 93.4 | 97.30 |

## 更快的评测

`tools/eval/` 里是上面评测用的合批策略服务器：仿真客户端的并发请求合成一次 `predict_action` 调用，不再一个一个
排队。见 [docs/faster/measurement.md](docs/faster/measurement.md#evaluation)。

## 上游 starVLA

基于 [starVLA/starVLA](https://github.com/starVLA/starVLA) 的 `99764f5`（MIT 许可）。原 README 保留为
[README.upstream.md](README.upstream.md)，作者信息和许可不变。
