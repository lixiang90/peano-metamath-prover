# Peano Metamath Prover

一个面向 `peano.mm` 的严格类型定理生成器与神经符号证明器。目前版本为
Beta：神经网络和搜索器只能提出候选动作，成功证明必须能够编译并由项目内
验证器重放；重要结果还应使用外部 Metamath 实现交叉验证。

## 特性

- 严格区分语法规则和逻辑断言：
  - `term`、`wff`、`BINOP` 等规则只构造和检查 AST；
  - 只有结论以 `|-` 开头的断言才能参与证明组合。
- 将证明对象统一表示为 `Γ ⊢ C`，支持部分前提消解和开放证明状态。
- 一阶合一、occurs check、精确类型检查和 Metamath `$d` 约束。
- 真空量词过滤、前提 subsumption、重复检测和数学结构质量评分。
- 原子形式 token 的 encoder-decoder Transformer 策略/价值模型。
- 传统动作枚举、辅助引理构造与 PUCT/MCTS 的混合搜索。
- 可恢复的 gzip 分片百万语料生成，以及约 100M 参数、2304 上下文训练。
- 未压缩 Metamath 证明导出、证书编译和按源文件顺序/作用域的内核回放。

## 项目结构

```text
formal/                  peano.mm 与保守数论定义
src/metamath_generator/  解析、合一、组合、质量控制与数据导出
src/neural_prover/       Transformer、MCTS、Scale 数据和训练
tests/                   内核、数论定义和神经符号回归测试
docs/                    架构、数论定义与 Scale 实验说明
```

训练数据、模型检查点和实验输出不会提交到 Git；默认写入 `outputs/`。

## 当前进展

截至 2026-08-11，仓库已经完成第一次百万级数据、GPT-2 Small 量级模型的
真实训练与深度压力测试：

- 生成 1,000,000 条可回放动作样本，其中训练集覆盖证明深度 1–16；
- 随机抽查 10,000 条动作，项目内符号环境回放 10,000/10,000 通过；
- 在 RTX 3090 上训练 104,130,543 参数模型至 5,000 optimizer steps，实际
  消费 40,000 个样本和约 3,099 万目标 token；
- 5,000 条隔离测试的 teacher-forced token accuracy 为 88.31%；
- 独立符号压力搜索生成到证明深度 19，神经压力测试覆盖到深度 17；
- 深度 14/15/16/17 的 token accuracy 约为
  80.37%/80.69%/80.03%/76.50%。

这些结果证明百万语料、100M 模型、长上下文和较深证明对象的工程管线可以
稳定运行，但**尚不能证明模型已经具备端到端解题能力**。当前完整参考动作
准确率仍为 0；更重要的是，动作复现本身只是模仿指标，同一状态可能存在多条
合法证明路径。下一阶段的主指标将改为：在固定搜索预算内能否找到完整证明、
消解全部开放前提，并输出由独立验证器接受的证书。

## 安装

仅使用符号生成器：

```bash
python -m pip install -e .
```

启用 Transformer 训练和 GPU 推理：

```bash
python -m pip install -e ".[neural]"
```

开发与测试：

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## 快速开始

生成经过质量过滤的定理、推理规则和开放证明状态：

```bash
python -m metamath_generator formal/peano.mm \
  --mode random --steps 5000 --seed 7 \
  --output-dir outputs/generated
```

输出包括：

- `closed_theorems.jsonl`
- `inference_rules.jsonl`
- `proof_states.jsonl`
- `quality_summary.json`

构建严格类型的神经策略语料：

```bash
python -m neural_prover build-corpus \
  formal/peano-number-theory.mm outputs/corpus \
  --seeds 7,11,19,23 --steps-per-seed 5000 \
  --max-state-tokens 384 --max-action-tokens 384

python -m neural_prover audit-corpus \
  formal/peano-number-theory.mm outputs/corpus \
  --output outputs/corpus/audit.json
```

构建去重的百万级分片语料：

```bash
python -m neural_prover build-scale-corpus \
  formal/peano-number-theory.mm outputs/corpus outputs/scale-1m \
  --train-examples 990000 --validation-examples 5000 \
  --test-examples 5000 --shard-size 20000 \
  --max-state-tokens 2304 --max-action-tokens 2304 --workers 8

python -m neural_prover audit-scale-corpus \
  formal/peano-number-theory.mm outputs/scale-1m \
  --sample-size 1000 --output outputs/scale-1m/audit.json
```

训练约 104M 参数、2304 上下文的模型：

```bash
python -m neural_prover train-scale \
  outputs/scale-1m outputs/model-104m \
  --max-steps 1000 --micro-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --initial-context-tokens 512 --context-warmup-steps 50 \
  --checkpoint-every 250 --device cuda
```

更多说明见：

- [系统架构](docs/architecture.md)
- [保守数论定义](docs/number-theory.md)
- [首次百万级训练与深度实验](docs/first-large-scale-run.md)
- [当前进展与研究路线](docs/progress-and-roadmap.md)
- [历史 Scale 基线](docs/scale-baseline.md)

## 下一阶段：闭环证明

计划中的真实证明流程是：

```text
目标与开放前提
    ↓
符号内核枚举合法候选动作
    ↓
神经策略排序，并可在输出前进行有限潜在思考
    ↓
Beam/MCTS 探索、执行与回溯
    ↓
所有开放前提消解
    ↓
编译证明 DAG 和 Metamath 证书
    ↓
项目内验证器与外部 Metamath 实现交叉验证
```

强化学习将只奖励实际通过验证的证明，而不是奖励复述训练集动作。连续潜在
思维只用于动作前的内部规划；所有改变形式状态的步骤仍必须是离散、可记录、
可验证的 Metamath 动作。详细阶段和验收标准见
[当前进展与研究路线](docs/progress-and-roadmap.md)。

## 数论语言边界

`formal/peano-number-theory.mm` 以显式保守定义增加素数、幂、有限序列、
有理数不等式和对数/Li 的有限证书。费马大定理、哥德巴赫猜想、素数定理及
黎曼猜想的 von Koch 等价形式使用 `statement` 类型命名，既不是公理，也
不是已证明定理；开放猜想永不作为训练成功标签。

## 可信边界

神经分数不是证明。系统只接受满足以下条件的结果：

1. 动作通过类型与 `$d` 约束检查；
2. 证明 DAG 能编译为 Metamath 证明；
3. 项目内验证器能够从原始形式库按声明顺序和活动作用域重放该证明。

当前验证器只支持未压缩证明，尚不能替代经过长期审计的通用 Metamath
验证器。发布或引用重要结论前，应再用 `metamath-exe` 等独立实现验证。

## 项目沿革

早期探索曾出现语法规则与定理混用、真空量化和开放前提变体过多、跨类型
替换，以及 Scale 语料约 1.21% 精确重复等问题。这些实现和实验资产没有进入
本发布目录；当前版本保留经过严格类型修复、质量过滤和全局去重的正式管线。

## 许可证

本项目按 [GPL-3.0](LICENSE) 发布。`formal/peano.mm` 保留 Robert Solovay
原始 GPL 版权声明，详见文件头和 [NOTICE](NOTICE)。
