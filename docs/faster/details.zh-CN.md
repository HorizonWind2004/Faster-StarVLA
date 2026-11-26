# Faster-StarVLA：细节

[返回 README](../../README.zh-CN.md) · [English](details.md)

这些数字来自冻结基线计时的那一次测量。之后的三次测量里基线本身更快（796–816 ms，而不是 905 ms），同样的比较得到 `all` 1.87×–1.96×、`exact` 1.56×–1.65×（[各次测量](measurement.md#the-judge-on-the-unified-code)）。

## 工作原理

### 三个档位

三档的区别在于允许改变什么，各自用相应的标准判定：

| 档位 | 增加的内容 | 与上游相比 | 评判等级 | 主配方 |
|---|---|---|:---:|---:|
| `exact` | 把工作挪出关键路径或干脆不做：输入处理放进 DataLoader 工作进程、不算用不到的词表 logits、掩码和位置表不再每次调用都重算、只在自检证明逐位一致的形状上合并注意力调用、精简 DeepSpeed 的记账开销 | 确定性模式下**逐位相同**：loss、每个梯度、更新量和优化器状态（在真实权重上核对过的框架，`predict_action` 输出的动作也相同） | N0 | 1.74× |
| `all` | `exact` 加上不降低任何运算精度、但可能改变浮点舍入顺序的改动：Inductor 编译层、SDPA 替代 flash-attention、任意形状的批量注意力、条件特征只投影一次、编译版 AdamW 更新 | **统计等价**：每个训练端点都在上游种子间波动的 3σ 以内 | N1 | 2.00× |
| `max` | `all` 加上精度设置：fp32 矩阵乘用 TF32（fp32 动作头）、PI0 / PI05 注意力的 softmax 用 bf16 | 统计等价，部分运算精度降低 | N2 | 没有额外收益（fp32 计算很少） |

确定性模式下，上游自己跑两次是逐位相同的；离开这个模式，上游两次运行通常在末位就已经不同（GPU 归约的顺序不固定），所以逐位比较只在
确定性模式下才有意义。评判等级沿用 [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent) 的定义：N0 逐位一致，
N1 同精度下的重排，N2 改变了数据类型或数学模式。

### 每条加速路径作用在哪里

全部代码在一个包里：[`starVLA/fast/`](../../starVLA/fast/__init__.py)，只在四个不属于任何框架的位置被调用：构建底座处
（`get_vlm_model`）、构建框架处（`build_framework`）、训练脚本拿到 DeepSpeed 引擎处、构建数据加载器处。
任何框架和底座接口里都没有开关，也没有一行加速代码。一条加速路径就是一个带 `apply(target) -> bool` 的模块加上
[`FAST_PATHS`](../../starVLA/fast/__init__.py) 里的一行；不适用时返回 `False`，日志里会列为“不适用”。

## 全部加速路径

| 加速路径 | 档位 | 作用位置 | 做什么、适用范围 |
|---|---|---|---|
| `worker_prep` | exact | 数据加载 | 提示词拼接、VLM processor 和旋转位置索引放到数据加载进程里做，不再占用训练进程；适用于所有把 batch 的图像和提示词交给底座接口（`build_qwenvl_inputs`）的框架，底座接口为 Qwen2.5-VL、Qwen3-VL、Cosmos-Reason2；会改写提示词的框架用一个函数登记改写方式 |
| `worker_prep_framework` | exact | 数据加载 | 同上，用于自己构建底座输入的框架（MiniCPM-RobotManip：它对 MiniCPM-V processor 的调用）；启用前先和框架自己的构建结果比对 |
| `video_prep` | exact | 数据加载 | 世界模型底座逐样本的图像缩放和归一化（Wan、Cosmos-Predict2）放到数据加载进程里做；工作进程里的 processor 先和训练进程里的比对 |
| `worker_examples` | exact | 数据加载 | PI0 / PI05：框架自己的样本预处理（图像缩放与填充、状态和动作填充）放到 DataLoader 工作进程里做 |
| `vision_attn` | exact | 视觉塔 | 每层一次批量注意力调用，取代逐张图片各算一次、每层还要和主机同步一次的做法；Qwen2-VL / Qwen2.5-VL / Qwen3-VL 视觉塔，注意力实现为 SDPA；只在当前 GPU 上自检确认批量调用逐位一致的图像形状上启用 |
| `vision_pos` | exact | 视觉塔 | 位置表按网格尺寸只建一次，不再每次前向都建；Qwen2-VL / Qwen2.5-VL / Qwen3-VL 视觉塔 |
| `attn_mask` | exact | 语言模型 | 因果 + 填充掩码由两次广播运算得到，取代四层嵌套的 `torch.vmap`；transformers 里用 `masking_utils` 构建掩码的底座（Qwen、Gemma、Llama、Phi 系列），且注意力实现为 SDPA |
| `hidden_only` | exact | 语言模型 | 不再计算用不到的词表 logits；任何带 LM head 的底座，调用时只要隐状态、不带 labels |
| `grad_norm` | exact | 优化器步 | 梯度裁剪的范数不再把每个梯度复制成 fp64（主配方上每步写入再读回 18 GB）；任何用 DeepSpeed ZeRO-2 训练的框架 |
| `allreduce` | exact | 优化器步 | 梯度桶原地 all-reduce；任何用 DeepSpeed ZeRO-2 训练的框架 |
| `sdpa` | all | 整个底座 | 用 flash-attention 加载的底座（或者只因某个子模型没有 SDPA 而用 eager 的底座，如 Gemma 4）切换到 PyTorch 的 scaled-dot-product attention：同一种注意力，换一个实现；下面针对 SDPA 的路径和编译层都依赖它。离开 flash-attention 后会逐个切片或窗口调用注意力、又不在 `vision_attn` 批量化范围内的视觉塔（MiniCPM-V 4.6）保留 flash-attention |
| `vision_attn_relaxed` | all | 视觉塔 | 对所有尺寸一致的图像形状做批量视觉注意力（末位舍入可能不同） |
| `compile_vision`、`compile_lm` | all | 视觉塔、语言模型 | 视觉 block 和解码器层在训练调用时用 Inductor 编译，并用 `gc.freeze()` 消除编译带来的垃圾回收停顿；适用于有 `model.visual.blocks` / `model.language_model.layers` 的底座（Qwen-VL 系列）以及其他 Hugging Face 多模态封装里的对应位置，且注意力实现为 SDPA；编译器处理不了的层保持 eager |
| `compile_world_model` | all | 世界模型 | 视频世界模型底座（Wan、Cosmos-Predict2）的 transformer 各层在训练调用时用 Inductor 编译 |
| `text_longest` | all | 世界模型 | Wan：4.7 B 的 UMT5 文本编码器只算到本批最长指令，不再算满 512 个填充位置（原版之后会把指令之后的位置清零）；只在编码器没有启用 dropout 时生效，也就是评测和推理。训练时原版让 UMT5 的 dropout 开着，这条路径不改动编码 |
| `frozen_no_grad` | all | 优化器 | Wan：UMT5 文本编码器和 VAE（65 亿参数）只在 `torch.no_grad` 里运行，上游却让它们保持可训练，ZeRO-2 因此为它们保存 fp32 副本和 Adam 状态，每步用全零梯度更新，结果逐位不变；这里在建优化器前把它们冻结。只在对它们的每次调用都处于 `no_grad` 中、且学习率 × weight decay 在 fp32 下会被舍入掉时启用；梯度裁剪的范数会有舍入差异，优化器检查点与不开这条路径的运行不能互换 |
| `condition_once` | all | 动作头 | 基于 starVLA 交叉注意力 DiT 的动作头（GR00T、逐层 flow matching、离散扩散）：框架为多个噪声水平复制的条件特征，每个样本只投影一次 |
| `compile_head` | all | 动作头 | 动作头的各层（GR00T、逐层 flow matching、离散扩散、LangForce、QwenDual 的 DiT 头，以及 Adapter 头的残差块）用 Inductor 编译；dropout 沿用 eager 的随机数 |
| `openpi_one_pass` | all | 视觉塔 | PI0 / PI05：一个 batch 的多个相机视角合成一次 SigLIP 调用，不再每个视角调用一次 |
| `unused_views` | all | 视觉塔 | PI0 / PI05：整个 batch 都没有的相机视角直接跳过，不再当作全黑图像编码 |
| `openpi_compile` | all | 整个底座 | PI0 / PI05：SigLIP 各层和 PaliGemma / 动作专家的联合层在训练调用时用 Inductor 编译 |
| `adamw_compiled` | all | 优化器步 | 所有用 DeepSpeed ZeRO-2 和 torch 融合 AdamW 训练的框架：每个 rank 的展平分区的更新编译成一个内核（B300 上 22.5 亿元素的分区，torch 融合内核 104.9 ms，编译版 9.4 ms） |
| `tf32` | max | 动作头 | fp32 矩阵乘使用 TF32，fp32 动作头的时间主要花在这上面。属于精度设置，所以不在 `all` 里 |
| `openpi_sdpa` | max | 整个底座 | PI0 / PI05：SigLIP 视觉塔和 PaliGemma / 动作专家的联合层改用融合的 scaled-dot-product attention，取代“矩阵乘、fp32 softmax、矩阵乘”；softmax 因此在 bf16 下计算 |
| `unit_clip` | 按名字 | 优化器步 | 裁剪系数恰好为 1 时不再遍历梯度去乘它；代价是每步要在主机上读一次系数（一次同步），换来省掉一遍 2.6 ms 的遍历 |
| `grad_cast` | 按名字 | 优化器步 | 分区梯度在展平的同时转成 fp32，一遍完成而不是两遍；在主配方上没有超出运行间波动的差别 |
| `reduce_to_owner` | 按名字 | 优化器步 | 梯度桶只归约到负责它的 rank；在两张卡上实测比默认做法慢 |

“按名字”的路径不属于任何档位，需要显式写出（`STARVLA_FAST=all,unit_clip`）。首版的逐路径环境变量
（`STARVLA_PREP_IN_WORKERS=1` 等）仍然有效。

## 其他框架和底座

starVLA 是一个包含很多策略和底座的框架。上面的主要数字来自一个配方（QwenOFT + Qwen3-VL-4B）。同一个设置用在其他
框架上的结果（硬件、数据、batch 相同，每一行都与同一作业里测的基线比较；共 23 个配方、18 个框架、9 个底座）：

| 框架 | 底座 | 配方里的注意力实现 | 基线 (ms/步) | `exact` | `all` | `max` | `exact` 逐位核对 |
|---|---|---|---:|---:|---:|---:|---|
| QwenOFT | Qwen3-VL-4B | SDPA | 801.6 | 489.8 (1.64×) | 292.2 (2.74×) | 305.5 (2.62×) | 评判程序：23,336 个张量；728 个梯度 + 动作 ✓ |
| QwenGR00T | Qwen3-VL-4B | SDPA | 880.4 | 605.2 (1.45×) | 399.1 (2.21×) | 392.6 (2.24×) | 960 个梯度 + 动作 ✓ |
| QwenPI | Qwen3-VL-4B | SDPA | 1293.8 | 1110.8 (1.16×) | 811.7 (1.59×) | 530.5 (2.44×) | 1,225 个梯度 + 动作 ✓ |
| QwenPI_v3 | Qwen3-VL-4B | SDPA | 1638.3 | 1389.5 (1.18×) | 849.8 (1.93×) | 519.7 (3.15×) | 1,376 个梯度 + 动作 ✓ |
| QwenDiscreteDiffusion | Qwen3-VL-4B | SDPA | 2512.9 | 2295.7 (1.09×) | 1884.9 (1.33×) | 566.9 (4.43×) | 1,220 个梯度 + 动作 ✓ |
| QwenAdapter | Qwen3-VL-4B | SDPA | 1052.3 | 767.9 (1.37×) | 553.0 (1.90×) | 555.8 (1.89×) | 1,226 个梯度 + 动作 ✓ |
| LangForce | Qwen3-VL-4B | SDPA | 1439.6 | 905.7 (1.59×) | 686.5 (2.10×) | 673.7 (2.14×) | 961 个梯度 + 动作 ✓ |
| QwenDual | Qwen3-VL-4B | SDPA | 1201.1 | 909.2 (1.32×) | 546.6 (2.20×) | 521.0 (2.31×) | 1,135 / 1,136 个梯度 + 动作（另 1 个上游自身两次运行也不同） ✓ |
| QwenOFT | Qwen2.5-VL-3B | SDPA | 1013.8 | 539.9 (1.88×) | 317.3 (3.20×) | — | 840 个梯度 + 动作 ✓ |
| QwenOFT | Qwen2.5-VL-3B | flash-attention | 877.6 | 645.6 (1.36×) | 466.5 (1.88×) | — | 840 个梯度 + 动作 ✓ |
| QwenGR00T | Qwen2.5-VL-3B | flash-attention | 861.9 | 692.9 (1.24×) | 405.3 (2.13×) | 405.2 (2.13×) | 1,072 个梯度 + 动作 ✓ |
| PI0 | PaliGemma (openpi) | eager | 639.2 | 591.5 (1.08×) | 290.3 (2.20×) | 288.9 (2.21×) | — |
| PI05 | PaliGemma (openpi) | eager | 670.2 | 626.8 (1.07×) | 308.3 (2.17×) | 315.9 (2.12×) | — |
| QwenGR00T | Qwen3.5-0.8B | flash-attention | 599.9 | 443.7 (1.35×) | 407.3 (1.47×) | 406.7 (1.48×) | — |
| QwenOFT | Qwen3.5-0.8B | flash-attention | 521.9 | 361.0 (1.45×) | 326.3 (1.60×) | 321.5 (1.62×) | — |
| Gemma4PI | Gemma-4-E2B (batch 4) | eager | 1080.1 | 965.0 (1.12×) | 594.3 (1.82×) | 581.3 (1.86×) | — |
| Gemma4GR00T | Gemma-4-E2B (batch 4) | eager | 1062.8 | 875.7 (1.21×) | 524.7 (2.03×) | 525.3 (2.02×) | — |
| MiniCPMGR00T | MiniCPM-V 4.6 | flash-attention | 867.9 | 570.4 (1.52×) | 519.2 (1.67×) | 527.1 (1.65×) | — |
| MiniCPMPI | MiniCPM-V 4.6 | flash-attention | 825.3 | 595.2 (1.39×) | 551.1 (1.50×) | 550.8 (1.50×) | — |
| MiniCPMRobotManip | MiniCPM-V 4.6 (RobotManip checkpoint; 10-D shapes from LIBERO) | eager | 1819.3 | 1532.9 (1.19×) | 498.4 (3.65×) | 499.7 (3.64×) | — |
| WanPI | Wan2.2-TI2V-5B (world model; head depth 30, upstream fails with 36) | SDPA | 2102.5 | 1755.7 (1.20×) | 958.9 (2.19×) | 725.3 (2.90×) | — |
| WanOFT | Wan2.2-TI2V-5B (world model) | SDPA | 1479.8 | 1136.2 (1.30×) | 540.2 (2.74×) | 538.0 (2.75×) | — |
| WanGR00T | Wan2.2-TI2V-5B (world model) | SDPA | 1611.8 | 1238.3 (1.30×) | 631.5 (2.55×) | 622.9 (2.59×) | — |

每一行都与同一个批处理作业里测的基线比较。同一节点上，不同作业之间同一配方的基线会相差最多 10%，所以不同作业的毫秒数不能直接比，行内的倍数可以比（见[各次测量](measurement.md#other-frameworks-and-backbones-step-time)）。

`exact` 的收益来自 VLM 输入处理和底座本身：这部分占比大的配方收益大（LangForce 每步构建两套 prompt：1.59×），PI0 / PI05 很小（1.08×），因为它们的一步基本就是 PaliGemma 的前向和反向。

flow-matching 动作头跑在 fp32。`condition_once`（属于 `all`）把框架为多个噪声水平复制的条件特征改成每个样本只投影一次。`max` 再让动作头的 fp32 矩阵乘使用 TF32；动作头每一层都要投影 VLM 隐状态的框架，剩下的时间大部分就是这些矩阵乘（QwenPI 812 → 531 ms；QwenPI_v3 850 → 520 ms；QwenDiscreteDiffusion 1885 → 567 ms）。TF32 是精度设置而不是运算顺序的变化，所以单列一档。评判程序在主配方上判定它统计等价；在 QwenPI 上按同样的方法检验（`all` 用 6 个种子、`max` 用 4 个种子），所有端点都在评判程序的 3σ 门限以内，最大的是 1.6σ（`max` 下的VLM 更新量），见[详情](measurement.md#the-statistical-gate-on-qwenpi)；其他框架在计时的 220 步里 loss 曲线与基线没有可分辨的差别（[数据](measurement.md#tf32)）。

Qwen2.5-VL 的配方默认用 flash-attention 加载底座。`exact` 保留 flash-attention，只启用与注意力实现无关的路径。`all` 先把底座切到 PyTorch 的 scaled-dot-product attention（`sdpa`），之后针对 SDPA 的路径和编译层才能生效：QwenOFT 878 → 646 (1.36×) → 467 (1.88×) ms。

Qwen2.5-VL 用 SDPA 时（bf16、B300），4 个全注意力视觉层（32 张图 × 256 个 patch，head 维度 80）没有通过逐位自检，所以 `exact` 只对 28 个窗口注意力层做批量；`all` 两种都做。

PI0 和 PI05 自带 PaliGemma，不走 `get_vlm_model`。它们的 `exact` 是 `worker_examples` 加两条 DeepSpeed 路径；`all` 再加 `unused_views`（LIBERO 只有这类模型所需三个相机视角中的两个，前缀从 816 个 token 缩到 560 个）和 `openpi_compile`；`max` 再加 `openpi_sdpa`。整个模型本来就跑在 bf16，所以 TF32 在这里没有作用。

新底座（第 13 次测量）：Wan2.2-TI2V-5B（WanGR00T、WanOFT、WanPI）、Qwen3.5-0.8B、Gemma-4-E2B 和 MiniCPM-V 4.6
（MiniCPMGR00T、MiniCPMPI、MiniCPM-RobotManip）都在上表里。在真实的 Qwen3.5-0.8B 上，`exact` 路径的 loss 和梯度逐位一致。
在它们身上起作用的主要是：

- **Wan：上游忘了冻结的子模型（`frozen_no_grad`）。** starVLA 的 Wan 接口只在 `torch.no_grad` 下调用 UMT5 文本编码器和 VAE
  （6.5 B 参数），但冻结它们的两行代码被注释掉了，于是 ZeRO-2 为它们保存 fp32 副本和 Adam 动量，每步用零梯度更新一次。
  在 LIBERO 的学习率和 weight decay 下，这次更新逐位不改变权重，所以冻结之后训练结果不变，只省下开销：每个 Wan 配方每步
  约 100 ms，显存峰值少 42–52 GiB（WanPI `all` 1.99× → 2.19×，`max` 2.53× → 2.90×）。同样的检查还找出了 ABot-M0 的 VGGT-1B
  空间模型（这里没有计时）。被冻结的子模型一旦在开启梯度时被调用，会直接报错，而不是悄悄比上游少训练一部分。
- **MiniCPM-V 4.6：逐块调用的视觉注意力。** 不用 flash-attention 时，它的视觉塔每层都要把分块长度读回 CPU，再对每个图像切片
  和每个 2 × 2 窗口各调用一次注意力（每步约 9,000 次）。`all` 档让这个视觉塔用 flash-attention（配方本来就用时保持，否则从
  SDPA 切过去），每层只调用一次：MiniCPM-RobotManip 每步 1,533 → 498 ms。`compile_vision` 不编译这种视觉塔。
- **Qwen3.5 和 Gemma 4。** `compile_lm` 不再编译带线性注意力层的语言模型（Qwen3.5），编译它会让第一步花 15 分钟、每步反而
  更慢；没有补齐的批次也拿到和其他批次一样的因果掩码，编译过的层不会在训练中途重新编译（Gemma 4 在 batch 4 时第 161 步遇到过）。
  Gemma 4 用 batch 4，因为它的 eager 基线放不下 16。

途中遇到的上游问题（计时时用让原版能跑起来的最小改动）：WanPI 设置的头部层数 30 已经不再被逐层动作头读取（它建了 36 层，
运行时报 `IndexError`；计时用 `diffusion_model_cfg.num_layers=30`）；MiniCPM-RobotManip 发布的动作头是 fp32，而 starVLA 的
DeepSpeed bf16 训练会把它存成 bf16（一个只用于计时的补丁在两棵代码树里都把动作头的输入转成同一精度），并且它训练用的 10 维
末端执行器数据这里没有（动作和状态补零到它要的形状，所以它的 loss 数值没有意义）；starVLA 的 Florence-2 接口只接受一个相机
视角而 LIBERO 配方给两个（只在底座的 loss 上做了距离检查）。`text_longest`（Wan）用于评测和推理。Qwen3.5、Gemma 4 和
MiniCPM-V 4.6 需要 transformers 5，上游固定的是 4.57。新配方都在同一个作业里和自己的基线比；在真实模型上，舍入距离检查覆盖了它们
的编译和动作头路径（WanGR00T、MiniCPM-RobotManip、Gemma4PI）；`frozen_no_grad` 和视觉塔切 flash-attention 由 CPU 测试（以及
上面的逐位论证）覆盖；真实模型上的评判（作业 73722，代码 8ac4405 对上游 99764f5）：WanGR00T 的 `exact` 逐位一致（两步后和评测后都是 17,204 / 17,204 个张量）；WanGR00T 的 `all,-frozen_no_grad` 和 MiniCPMGR00T 的 `all` 通过统计检验（最大偏差 1.91σ、0.85σ；学习率 ×1.02 的对照分别在 17.6σ、22.4σ 被拒绝）。WanGR00T 的 `all` 在种子 0 上有一个端点（VLM 更新幅度）到了 3.05σ，小于同一端点上上游自己对自己的 4.78σ；合并 4 个种子后每个端点都在 0.90σ 以内，判为等价，但这个端点单种子的噪声比 6 个种子估出的 σ 大。MiniCPMGR00T 的上游自己就不能逐位复现（3,175 / 16,436 个张量），所以它的 `exact` 档无法用逐位检查判定。没有实测：Molmo2、VILA，以及真实的 Cosmos-Reason2 权重
（它的接口只在 CPU 测试里用 Qwen3-VL 结构跑过）；QwenFast（它把标签文本交给接口，`worker_prep` 遇到这种调用会留给训练进程
处理）、InternVLA-M1 和 ABot-M0；Cosmos-Predict2 和纯视觉框架；co-training 与 VLM 训练脚本（它们调用的是同样的入口）。

## 验证

所有结论都来自优化开始之前就构建并冻结的评判程序（[judge/](judge/)，
[判定方式](measurement.md#the-judge)）。它依次执行四道门，判定结果带一个等级：

| 门 | 检查内容 |
|---|---|
| 0 结构 | 训练入口的初始化部分与上游的语法树相同；数据管线产出的样本和前几个 batch 与上游逐位相同 |
| 1 收益 | 真实入口的全新进程、冷编译缓存、预热 20 步，然后两个 100 步的窗口；按“参照 → 候选 → 参照”的顺序跑，加速比必须超过 `max(1.02, 1 + 2 × 参照波动)` |
| 2 逐位 | 确定性模式下两个 2 步用例，两张卡都记录：先核对精确项（随机数状态、学习率、优化器和 DeepSpeed 设置、可训练参数、数据类型、VLM 输入摘要），再逐位比较 loss 以及每个参数的 fp32 梯度、更新量和 Adam 状态。全部相同即 **PASS N0** |
| 3 统计 | 只在第 2 道门不逐位一致时执行：固定种子跑 60 步轨迹，七个端点都在上游种子间波动（6 个种子）的 3σ 以内即 **PASS N1**；若改变了数据类型或数学模式（TF32、低精度归约、SDPA 后端）则为 **N2** |

优化开始前已经验证过评判程序会拒绝该拒绝的情况（改了训练设置、改了裁剪阈值、VLM 收不到梯度、所有学习率乘 1.02），并放行等价的重排。

**`STARVLA_FAST=exact`。** 打开适用于这个配方的七条逐位一致的加速路径，在确定性模式下跑两个 2 步训练用例（其中一个在两步之间插入配方里的周期性
评估），两张卡都记录：23,336 个张量（loss，以及全部 729 个参数的 fp32 梯度、更新量和两个 Adam 状态）与上游
逐位相同；学习率、随机数状态、喂给 VLM 的输入摘要也都相同。

**`STARVLA_FAST=all`。** 编译层会改变浮点舍入，所以完整集合用统计方式比较：确定性模式下、与上游同一个种子，跑一条 60 步的
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

**在当前代码上。** 上面的检查是在首版上做的，当时加速路径还没有统一到 `STARVLA_FAST`，也没有扩展到其他框架。在当前代码上，评判程序再次确认 `STARVLA_FAST=exact` 逐位一致：23,336 个张量中 23,336 个相同；之后跑过这项检查的6次测量都是如此，最后一次用的就是发布的这个提交。`STARVLA_FAST=all` 的三进程检查在之后的测量里都没有给出结论：基线比冻结计时快，超出了允许的范围，评判程序按规则拒绝判定（INVALID）。它自己测到的加速比是 1.87×–1.96×。代理式检查（一次候选运行与冻结计时比较，再跑逐位门和统计门）在发布的提交上对 `all` 判定通过（等级 N1，各端点在 1.77σ 以内；每步 340 ms，是冻结计时 906.9 ms 的 2.66×）；在较早的一个提交上，对 `all,tf32`（即 `max` 在这个配方上启用的全部路径）也通过（等级 N2，1.41σ 以内）。[详情](measurement.md#the-judge-on-the-unified-code)。

**其他框架的逐位核对。** 评判程序只针对主配方；此外，上表里凡是权重在这里的框架，都在真实权重上把 `exact` 与基线逐位比较过：
确定性模式下的一个训练步（loss 和每个梯度）加一次 `predict_action` 调用（上表最后一列；[全部核对](measurement.md#other-frameworks-and-backbones-bitwise-checks-on-real-weights)）。
各条只改变舍入的路径会让一个训练步偏离多少，以及把 SDPA 换成 eager 注意力本身会带来多大偏离，也[在那里有实测](measurement.md#how-far-a-rounding-level-path-moves-one-training-step)。

**LIBERO 评测。** 用全部加速路径完整训练一次（30,000 步，种子 42，2 × B300，全局 batch 32），按 starVLA 的 LIBERO 协议
评测（每个套件 500 个 episode），并在同一条评测管线里评测官方发布的 checkpoint。官方 checkpoint 由 starVLA 作者在
8 张 GPU 上训练（全局 batch 128，取第 50,000 步）；这里只是检查加速后的代码能训出可用的策略，不是同种子的对照。这 30,000 步
实际用时 3 小时 51 分钟（含配方里的周期性评估和三次 checkpoint 保存，平均每步 462 ms）；按冻结的上游单步耗时估算，
只算训练步就要 7.6 小时（没有实际跑）。

| Success rate (%) | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|
| starVLA Qwen3-VL-OFT, published | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| Released checkpoint (`StarVLA/Qwen3-VL-OFT-LIBERO-4in1`, step 50,000), our evaluation | 98.6 | 100.0 | 99.4 | 96.6 | 98.65 |
| **Faster-StarVLA run** (all fast paths, 30,000 steps on 2 GPUs) | 98.8 | 98.6 | 98.4 | 93.4 | 97.30 |

## 常见问题

**会改变训练结果吗？** `exact` 不会：确定性模式下，一个训练步记录的每个张量都逐位相同；在真实权重上核对过的框架，
`predict_action` 输出的动作也相同。`all` 做的是同样精度下的同样运算，只是编译后的内核舍入顺序不同；评判程序用统计方式比较，
结果在上游自身的种子间波动之内，用它完整训练的 LIBERO 策略也能正常工作（见上）。`max` 会降低部分运算的精度（TF32、bf16 softmax），
评判等级为 N2。

**怎么关掉某一条，或者单独加一条？** 写在档位后面：`STARVLA_FAST=all,-compile_lm` 去掉一条，`STARVLA_FAST=exact,compile_head`
加一条。写错名字会直接报错，不会被悄悄忽略。启动日志会按入口列出实际生效的路径和对当前模型不适用的路径。

**推理能加速吗？** 给策略服务器设置同一个变量即可。编译层、`condition_once` 和 DeepSpeed 相关路径只作用于训练（需要梯度的调用）；
`predict_action` 里其余路径照样生效，比如不算 LM head、直接构建掩码、批量视觉注意力、位置表缓存、`sdpa` 和 `text_longest`。
`tools/eval/` 另外提供一个把模拟器并发请求合批的策略服务器（[见下](#更快的评测)）。

**第一步会变慢吗？** `all` 档的前几个训练步要用 Inductor 编译各层。PI0 在编译缓存为空时 445 秒后跑到第一步，用上一次运行
留下的缓存则是 165 秒；缓存在多次运行之间保留（`TORCHINDUCTOR_CACHE_DIR`）。上面的计时都从预热 20 步之后开始，
计时窗口里出现编译的运行会被判为无效。

**显存呢？** `all` 档更低：主配方上每卡峰值 66.5 → 59.0 GiB，主要来自编译解码器层的那一轮。各配方的数字见
[测量细节](measurement.md#other-frameworks-and-backbones-step-time)。

**哪些底座需要 transformers 5？** Qwen3.5 和 Gemma 4（上游固定 4.57）。加速路径在两个版本上都能用；每个新底座的进展见
[其他框架和底座](#其他框架和底座)。

**其他 GPU 呢？** 这里的一切都在 B300 上测得（torch 2.9.1、CUDA 12.9）。`vision_attn` 会在当前 GPU 上做逐位自检，按设备决定
合并哪些形状。其他 GPU 上的加速比和逐位一致性都没有验证过；[judge/](judge/) 说明了怎样在别的机器上重跑评判程序。

## 更快的评测

`tools/eval/` 里是上面评测用的合批策略服务器：仿真客户端的并发请求合成一次 `predict_action` 调用，不再一个一个
排队。见 [measurement.md](measurement.md#evaluation)。
