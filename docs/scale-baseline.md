# Scale 基线

本页记录早期可复现的 1,000-step 工程基线，不表示模型已经充分收敛，也不再
代表当前最佳训练结果。5,000-step 百万语料实验见
[首次百万级训练与深度实验](first-large-scale-run.md)，后续闭环解题指标见
[当前进展与研究路线](progress-and-roadmap.md)。

2026-09-04 核对时保留以下历史数字；之后还有
[RTX 5090 的10,000步实验](first-rtx5090-closed-loop-run.md)和
[PA+ 小型 GPU 验证](pa-plus-neural-training.md)，它们不是本页同一实验的续写。

## 数据

- 总量：1,000,000
- train / validation / test：990000 / 5000 / 5000
- gzip JSONL 分片：52
- 压缩大小：约 0.43 GiB
- 最大状态/动作长度：2304 / 2304
- 超过 2048 token：33106
- 全量唯一 state/action ID：1000000 / 1000000
- 项目内严格验证器抽查：1000 / 1000 合法

该抽查结果尚未等同于外部交叉验证；发布数据集时应另行报告
`metamath-exe` 等独立实现的证书通过率。

## 模型

- 参数量：104130543
- `d_model=768`
- attention heads：12
- encoder / decoder：6 / 6
- FFN：3072
- CUDA bf16、梯度检查点、micro-batch 1、梯度累积 4

## RTX 3090 首轮训练

- optimizer steps：1000
- 训练样本：4000
- 训练时间：约 230.5 秒
- 吞吐：17.36 examples/s
- target token 吞吐：约 11.5k/s
- 峰值 CUDA 分配显存：2.35 GiB
- 超过 2048 token 的真实反向传播：通过

256 条独立测试样本：

| 指标 | 结果 |
|---|---:|
| loss | 1.2303 |
| perplexity | 3.4114 |
| token accuracy | 47.43% |
| exact action accuracy | 0% |

混合 MCTS 小规模严格评估：

| 难度 | 内核认证 |
|---|---:|
| easy | 2/2 |
| medium | 2/2 |
| hard | 1/2 |
| 总计 | 5/6 |

exact-action 仍为 0，并且混合 MCTS 同时使用传统动作枚举。因此 `5/6` 不能
全部归因于神经策略。此基线证明百万数据、100M 模型和 2304 上下文管线可
运行；证明规模效应需要更长训练、纯符号基线对照和多随机种子置信区间。

这里的 exact-action 是“逐 token 完全复现参考动作”，不等同于实际解题率；
小规模 MCTS 结果也没有完成纯符号、启发式、神经和混合策略的严格消融。后续
不再用这两个数字单独宣称证明能力。
