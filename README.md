# Peano Metamath Prover — HTPS 增强版

这是 `peano-metamath-prover` 仓库的 `htps` 分支，在保留增强 PA 形式内核、认证
证书、中间引理动作和隐式连续思维链的基础上，引入 HTPS 与生成式 PA+ 支持：

- 认证前向证明 DAG 数据生成，并按递归证明骨架隔离数据集；
- 真正的 AND/OR 超图搜索、节点共享、PUCT 和软 critic 回传；
- 中间引理的受保护顺序超边：先证明 `Γ ⊢ L`，再开放 `Γ,L ⊢ G`；
- 只从最小已认证证明提取策略目标，预算耗尽状态仍按删失处理；
- latent policy/value/lemma/halt 的监督训练与同步在线闭环；
- 仅以最终 Metamath 证书重放成功作为 solve 指标。

三个包共同维护在本仓库：`metamath_generator` 提供形式内核和 PA+ 生成器，
`neural_prover` 提供基础模型与 PA+ 训练/推理，`htps_prover` 提供超图搜索和 latent
闭环。无需相邻的 `peano-metamath-prover-htps` 文件夹。

文档于 2026-09-04 按代码版本 `b911e9c` 核对。以下命令在本仓库根目录执行；
`bash` 代码块的多行续行使用 `\`，PowerShell 应改为反引号或合并为一行。

## HTPS 快速开始

```bash
python -m pip install -e ".[neural,dev]"

# 1. 从头训练：生成 forward DAG、policy 数据和 lemma 数据
peano-htps generate formal/peano.mm outputs/htps-data \
  --steps 5000 --seeds 7,11,19,23 --max-proof-depth 12

# 2. 新建约 100M 参数量级的 latent+lemma 模型（精确参数量随词表变化）
peano-htps init-model outputs/htps-data/tokenizer.json \
  outputs/latent-initial.pt

# 3. 用 forward DAG 监督训练
peano-htps train-supervised outputs/latent-initial.pt \
  outputs/htps-data/tokenizer.json outputs/htps-data/policy_train.jsonl \
  outputs/latent-sft.pt --lemmas outputs/htps-data/lemma_train.jsonl

# 4. 同步 HTPS 搜索/训练闭环
peano-htps closed-loop outputs/latent-sft.pt \
  outputs/htps-data/tokenizer.json outputs/htps-data/policy_train.jsonl \
  formal/peano.mm outputs/closed-loop --iterations 2 --examples 32

# 5. 在隔离 split 上进行证书口径评估
peano-htps evaluate outputs/closed-loop/checkpoint-002.pt \
  outputs/htps-data/tokenizer.json outputs/htps-data/policy_test.jsonl \
  formal/peano.mm outputs/evaluation.json --limit 32
```

算法、数据格式、可信边界和已知限制见
[`docs/htps-design.md`](docs/htps-design.md)。

若从旧 checkpoint 启动，第1步须追加 `--base-tokenizer BASE_TOKENIZER.json`，
第2步改用 `peano-htps init-latent BASE.pt BASE_TOKENIZER.json outputs/latent-initial.pt outputs/htps-data/tokenizer.json`。
不得把从头生成的 tokenizer 与旧 checkpoint 任意混配。HTPS 当前认证口径为项目内
证书重放，尚未接入官方 Metamath 的强制验证 CLI 选项。

PA+ 的目录、有界实例和35目标生成参数已接入 `neural_prover build-corpus`；
尚未接入 `peano-htps generate` 或完整 Scale 路径。PA+ 实验请使用下方专门示例。

这是面向 `peano.mm`、数论定义库和 `peano-pa-plus.mm` 的严格类型定理生成器与
神经符号证明器。目前版本为
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
- PA+ 定义上下文、目标标签提示、类型化替换槽位与可认证定义桥搜索宏。
- PA+ 有界自然数实例、目标引导生成、批内候选损失与分类型加权采样。
- 对内核合法有限候选集合进行索引打分的结构化策略头。
- 可验证的中间引理 cut 动作：先证明引理，再将其用于最终目标。
- 可选的隐式连续向量思维链与学习停止头；旧模型文件和检查点保持兼容。
- 传统动作枚举、辅助引理构造与 PUCT/MCTS 的混合搜索。
- AlphaZero 式 MCTS 回放；预算耗尽按删失处理，不伪造负奖励。
- 可恢复的 gzip 分片百万语料生成，以及约 100M 参数、2304 上下文训练。
- 未压缩 Metamath 证明导出、项目内核回放和外部 Metamath 交叉验证。

## 项目结构

```text
formal/                  peano.mm、保守数论定义与生成式 PA+ 目录
src/metamath_generator/  解析、合一、组合、质量控制与数据导出
src/neural_prover/       Transformer、MCTS、Scale 数据和训练
src/htps_prover/         forward DAG、HTPS、latent 训练与在线闭环
tests/                   内核、数论定义和神经符号回归测试
docs/                    架构、数论定义与 Scale 实验说明
```

训练数据、模型检查点和实验输出不会提交到 Git；默认写入 `outputs/`。

## 当前进展

最新工程进展是 2026-08-31 的 PA+ 本地 CUDA 验证：68个高层定义、35个非逻辑目标，
136条展开/折叠桥和有界实例已接入基础神经管线。RTX 3060 Laptop GPU 上用616例
语料训练64维、2层 encoder/decoder 模型3轮，验证损失从5.9908降至3.8880；一个
定义桥实例经混合搜索闭合，证书通过项目内和官方 Metamath 双验证。这是接线验证，
不是35目标解题率实验，也不是百万级 PA+ 训练。详见
[PA+ 神经训练与闭环推理](docs/pa-plus-neural-training.md)。

以下保留 2026-08-24 的历史规模结果，来自数论库而非新增 PA+ 语料：

- 生成 1,100,000 条可回放动作样本，训练/验证/测试为
  1,080,000/10,000/10,000；
- 随机抽查 10,000 条动作，项目内符号环境回放 10,000/10,000 通过；
- 在 RTX 5090 上训练 105,312,496 参数模型至 10,000 optimizer steps，实际
  消费 80,000 个样本和约 7,304 万目标 token；
- 5,000 条隔离测试的 teacher-forced token accuracy 为 92.26%，完整参考动作
  准确率仍为 0；
- 9 个固定研究目标、四策略同预算的闭环评估中，uniform/heuristic/neural/hybrid
  分别认证 0/9、4/9、0/9、3/9；全部 7 个成功证书通过官方 Metamath；
- 50 条可验证 replay 的候选头更新没有提高 solve rate，但将纯神经搜索耗时从
  147.4 秒降至 63.6 秒（2.32 倍加速）。

这些结果证明百万语料、100M 模型、长上下文、候选头、MCTS 回放和双验证工程
闭环可以稳定运行，但**尚不能证明学习策略改善了解题能力**。当前小样本中神经
策略没有解出目标，P1/P2 的 solve-rate 验收未通过；下一步需要扩大 replay，并在
37 个泄漏合格目标、三个 seed 上完成正式报告。

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

上述安装仅覆盖不依赖 PyTorch 的测试；完整神经/HTPS 回归应安装 `.[neural,dev]`。
本地已有 PyTorch 时，也可在 PowerShell 中设 `$env:PYTHONPATH='src'` 后运行
`python -m unittest discover -s tests`，无需重复安装。

## 快速开始

生成经过质量过滤的定理、推理规则和开放证明状态：

```bash
python -m metamath_generator formal/peano.mm \
  --mode random --steps 5000 --seed 7 \
  --output-dir outputs/generated
```

对扩展 PA+ 目录启用定义桥、小自然数闭式实例、35目标反向引导和定义包装限流：

```bash
python -m metamath_generator formal/peano-pa-plus.mm \
  --mode random --steps 5000 --seed 7 \
  --definition-catalog formal/pa-plus-definitions.json \
  --bootstrap-definitions --definition-coverage-weight 3 \
  --bounded-nat-max 2 --ground-instances-per-predicate 1 \
  --target-guidance-weight 4 \
  --max-definition-only-search-per-predicate 1 \
  --output-dir outputs/pa-plus-guided
```

算法、基线和严格边界见
[PA+ 随机定理生成](docs/pa-plus-random-generation.md)。

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

PA+ 神经语料可以直接复用定义目录、闭式实例和35目标反向引导：

```bash
python -m neural_prover build-corpus \
  formal/peano-pa-plus.mm outputs/pa-plus-corpus \
  --definition-catalog formal/pa-plus-definitions.json \
  --bootstrap-definitions --definition-coverage-weight 3 \
  --bounded-nat-max 2 --ground-instances-per-predicate 1 \
  --target-guidance-weight 4 \
  --max-definition-only-search-per-predicate 1 \
  --max-proof-depth 8 --max-ast-depth 64 --max-variables 24 \
  --max-state-tokens 1024 --max-action-tokens 512
```

词表、候选头训练、旧 checkpoint 升级和推理时的保守定义桥见
[PA+ 神经训练与闭环推理](docs/pa-plus-neural-training.md)。

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
- [HTPS 数据、训练与推理](docs/htps-design.md)
- [保守数论定义](docs/number-theory.md)
- [PA+ 分层定义库](docs/pa-plus-definitions.md)
- [PA+ 随机定理生成](docs/pa-plus-random-generation.md)
- [PA+ 神经训练与闭环推理](docs/pa-plus-neural-training.md)
- [形式系统演进设计（含未实现部分）](docs/formal-system-evolution.md)
- [首次百万级训练与深度实验](docs/first-large-scale-run.md)
- [首次 RTX 5090 规模训练与闭环实验](docs/first-rtx5090-closed-loop-run.md)
- [中间引理与连续潜在思维](docs/lemma-and-latent-reasoning.md)
- [当前进展与研究路线](docs/progress-and-roadmap.md)
- [历史 Scale 基线](docs/scale-baseline.md)

## 已实现的闭环证明与待完成验收

P0 可信评估、P1 有限候选策略头和 P2 AlphaZero 回放的工程路径已经实现；正式
研究验收仍需固定 benchmark 的三种子实验。构建公开 benchmark 时应同时传入
训练语料，参考规则只写到私有 sidecar：

```bash
python -m neural_prover build-benchmark \
  formal/peano-number-theory.mm outputs/benchmark-v2.json \
  --training-corpus outputs/corpus \
  --reference-output outputs/benchmark-reference.private.json

python -m neural_prover evaluate-mcts \
  outputs/model/final.pt outputs/corpus outputs/benchmark-v2.json \
  formal/peano-number-theory.mm outputs/baselines.json \
  --policies uniform,heuristic,neural,hybrid --seeds 17,19,23 \
  --external-verifier /path/to/metamath \
  --require-external-verification
```

`METAMATH_EXECUTABLE` 环境变量也可指定外部验证器。若要求外部验证但程序不存在、
超时或输出不明确，结果不会被计为 certified。

在不覆盖旧文件的前提下，把既有检查点升级为中间引理 + 连续潜在思维版本：

```bash
python -m neural_prover init-latent \
  outputs/model/final.pt outputs/corpus/tokenizer.json \
  outputs/model-latent/initial.pt outputs/model-latent/tokenizer.json

python -m neural_prover prove-decomposed \
  outputs/model-latent/initial.pt outputs/model-latent/tokenizer.json \
  outputs/benchmark-v2.json CASE_ID formal/peano-number-theory.mm \
  --simulations 100 --max-depth 20 --branching 24
```

升级会保留全部旧 token ID 及对应权重，只追加引理动作 token 和新模块。新增权重
仍需用认证 replay 训练；该工程原型本身不构成解题率提升的实验结论。设计与验收
边界见[中间引理与连续潜在思维](docs/lemma-and-latent-reasoning.md)。

已实现的证明流程是：

```text
目标与开放前提
    ↓
符号内核枚举合法候选动作
    ↓
神经策略排序，并可在输出前进行有限潜在思考
    ↓
    Best-first / MCTS / HTPS 探索、执行与回溯
    ↓
所有开放前提消解
    ↓
编译证明 DAG 和 Metamath 证书
    ↓
项目内验证器与外部 Metamath 实现交叉验证
```

闭环回放只把实际通过验证的完整证明标为成功；预算耗尽不伪造负价值标签。连续潜在
思维只用于动作前的内部规划；所有改变形式状态的步骤仍必须是离散、可记录、
可验证的 Metamath 动作。详细阶段和验收标准见
[当前进展与研究路线](docs/progress-and-roadmap.md)。

## 数论语言边界

`formal/peano-number-theory.mm` 以显式保守定义增加素数、幂、有限序列、
有理数不等式和对数/Li 的有限证书。费马大定理、哥德巴赫猜想、素数定理及
黎曼猜想的 von Koch 等价形式使用 `statement` 类型命名，既不是公理，也
不是已证明定理；开放猜想永不作为训练成功标签。

`formal/peano-pa-plus.mm` 在其上增加由 JSON 目录机械生成的 68 个高层定义和
35 个闭合目标公式。编译器自动生成 `$d` 条件并拒绝自由变量泄漏、递归或前向
定义依赖；具体接口和重新生成命令见
[PA+ 分层定义库与生成接口](docs/pa-plus-definitions.md)。
神经侧会把定义目录、目标提示和有界闭项编码成结构化上下文，并把生成期定义桥
内联为原始 `df-*` 证明；具体设计与本地 GPU 验证见
[PA+ 神经训练与闭环推理](docs/pa-plus-neural-training.md)。

## 可信边界

神经分数不是证明。系统只接受满足以下条件的结果：

1. 动作通过类型与 `$d` 约束检查；
2. 证明 DAG 能编译为 Metamath 证明；
3. 项目内验证器能够从原始形式库按声明顺序和活动作用域重放该证明。

当前项目内验证器只支持未压缩证明，尚不能替代经过长期审计的通用 Metamath
验证器。`neural_prover evaluate` / `evaluate-mcts` 可调用外部 Metamath；发布或
引用这些评估结果时应使用 `--require-external-verification` 将其设为强制认证门槛。
该选项不是所有子命令共有的能力，HTPS 的外部强制验证仍待接入。

## 项目沿革

早期探索曾出现语法规则与定理混用、真空量化和开放前提变体过多、跨类型
替换，以及 Scale 语料约 1.21% 精确重复等问题。这些实现和实验资产没有进入
本发布目录；当前版本保留经过严格类型修复、质量过滤和全局去重的正式管线。

## 许可证

本项目按 [GPL-3.0](LICENSE) 发布。`formal/peano.mm` 保留 Robert Solovay
原始 GPL 版权声明，详见文件头和 [NOTICE](NOTICE)。
