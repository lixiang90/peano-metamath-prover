# 统一训练与评估指南

[文档导航](README.md) · [English overview](../README.md) · [中文概览](../README-zh.md)

2026-09-04，PA+、MCTS 与 HTPS 统一在 `main` 维护。这里集中可运行入口和兼容边界；
各算法细节仍留在专题文档，不把历史实验参数当作新的规模训练推荐。

## 1. 安装与兼容

在仓库根目录执行，Python 3.10+。下文为 Bash 续行语法；PowerShell 请将 `\` 改成
反引号或合并为一行。已有 PyTorch 时可先设 `$env:PYTHONPATH='src'`，直接运行模块命令。

```bash
python -m pip install -e ".[neural,dev]"
python -m unittest discover -s tests
```

包名恢复为 `peano-metamath-prover`。若环境曾安装旧 `peano-metamath-prover-htps`
发行包，建议新建虚拟环境安装；若沿用旧环境，先执行下方迁移，再重新安装统一包。
两者提供相同模块和脚本，不能长期混装。该卸载不删除仓库内数据和 checkpoint。

```bash
python -m pip uninstall peano-metamath-prover-htps
python -m pip install -e ".[neural,dev]"
```

| 入口 | 等价模块命令 | 用途 |
| --- | --- | --- |
| `metamath-generate` | `python -m metamath_generator` | 共享符号生成器 |
| `pa-plus-compile` | `python -m metamath_generator.definitions` | PA+ 定义编译与审计 |
| `peano-neural-prover` | `python -m neural_prover` | 基础训练、MCTS、Scale、latent 升级 |
| `peano-htps` | `python -m htps_prover` | HTPS DAG 数据、latent 训练、超图闭环 |

三个源码包、既有 CLI 和 checkpoint 文件路径均保留，不要求移动旧输出。
模型必须搭配自己的 tokenizer；同样的符号集合不保证相同 token ID。基础模型转
latent 用显式 `init-latent`；PA+ 词表扩容用 `upgrade-pa-plus`，输出到新路径。
新增权重需要训练；基础 `train --checkpoint ... --checkpoint-tokenizer ...` 现可
加载升级后的基础模型，以新优化器微调。模型结构取自 checkpoint，不用新建模型的默认值。
旧 PA+ `legacy-unordered` 桥动作不能沿用，须按
[认证记录](pa-plus-certification.md)重新生成、审计和训练。

动作与状态编码修复后，监督语料、Scale 和搜索回放统一使用类型化的规则绑定槽位。
MCTS / best-first 的模型输入和回放键包含完整、有序的目标队列；重复的 `<GOAL>` 段
表示剩余义务，`<GOAL> <LEMMA> ... <END_LEMMA>` 表示证明完成后才激活的引理。
单目标输入与词表 token ID 保持不变。完整状态使用 `proof_state_tokens` /
`proof_state_from_tokens`，动作解码可直接接收 `ProofState`；`as_theorem()` 仅用于
当前目标的定理接口，不得作为多目标状态的序列化结果。

修复前生成的 Scale 语料及 MCTS / HTPS 回放应在新的输出目录重新生成。旧多目标回放
已丢失后续目标，不能只转换 token 恢复；已训练权重也不会自动修复，需要重新评估或
微调。旧词表无法表达待激活引理时，搜索退回符号评分，并跳过无法完整编码的回放状态。
MCTS 成功路径的价值标签与 backup 一致，按剩余步骤扣罚；Scale 在分配模型之前根据
实际记录检查课程长度、末步长上下文及验证范围，无匹配样本或空 split 会明确报错。

## 2. 工作流选择与当前接线

| 能力 | 基础 / MCTS | HTPS | Scale |
| --- | --- | --- | --- |
| 共享 PA / 数论内核 | 已支持 | 已支持 | 已支持 |
| PA+ 目录、定义桥、有界项、35目标引导生成 | `build-corpus` 已支持 | `generate` 已支持，自动全量动作审计 | PA+ 桥模板实例化已接入；闭式实例另行重采样 |
| 数据形式 | 单步动作语料 | 按证明骨架与相同根状态连通分组的 policy / lemma 数据 | gzip 动作分片 |
| 训练 | 动作/价值 + 可选批内候选损失；支持基础 checkpoint 微调 | latent 动作/价值/ponder、类型采样、可选批内候选损失；replay 候选训练 | 可恢复长上下文监督训练 |
| 搜索回放 | MCTS 访问分布 + 价值 | 最小认证证明策略边 + 软 critic | 不是第三种搜索算法 |
| 强制外部认证入口 | `evaluate` / `evaluate-mcts` | `evaluate` / `collect` / `closed-loop` | 训练诊断不等于证明认证 |

PA+ 是共享形式层，不是某个后端专属功能。两个生成入口现均已适配，但不能
把基础 `train.jsonl` 直接替换 HTPS `policy_train.jsonl`，也不能把普通 checkpoint
当作 latent checkpoint。HTPS 从生成、审计、latent SFT 到评估均按配套 tokenizer
配置 PA+。新增完整示例和实测见 [PA+ 数据与训练贯通](pa-plus-htps-training.md)。

## 3. PA+：生成、全量审计、小模型训练

这是本地工程验证配置，不是百万级训练，也不测量35个目标的解题率。选择新的
输出目录保存修复后的语料，训练前查看审计报告并确认没有无效动作。

```bash
python -m neural_prover build-corpus formal/peano-pa-plus.mm \
  outputs/unified-pa/corpus --seeds 7,11 --steps-per-seed 300 \
  --max-proof-depth 8 --max-ast-depth 64 --max-variables 24 \
  --definition-catalog formal/pa-plus-definitions.json \
  --bootstrap-definitions --definition-coverage-weight 3 \
  --bounded-nat-max 2 --ground-instances-per-predicate 1 \
  --target-guidance-weight 4 --max-definition-only-search-per-predicate 1 \
  --max-state-tokens 1024 --max-action-tokens 512

python -m neural_prover audit-corpus formal/peano-pa-plus.mm \
  outputs/unified-pa/corpus --output outputs/unified-pa/audit.json

python -m neural_prover train outputs/unified-pa/corpus outputs/unified-pa/model \
  --device cpu --epochs 3 --batch-size 16 --d-model 64 --layers 2 \
  --learning-rate 0.0005 --candidate-loss-weight 0.35
```

GPU 可用时将 `--device cpu` 改为 `--device cuda`。模型输出 `best.pt`、`final.pt`
和 `metrics.json`，tokenizer 留在 `corpus/tokenizer.json`，不会自动复制到模型目录。
细节见 [PA+ 神经管线](pa-plus-neural-training.md)。

## 4. 基础模型 / MCTS：独立 benchmark 与四策略评估

下面单独构造数论库基线，保证模型、语料和数据库相配；不要混用上一节的 PA+ 模型。
为基准检查训练重叠，公开 benchmark 与私有参考 sidecar 分开保存。

```bash
python -m neural_prover build-corpus formal/peano-number-theory.mm \
  outputs/unified-mcts/corpus --seeds 7,11 --steps-per-seed 500 \
  --max-state-tokens 384 --max-action-tokens 384

python -m neural_prover audit-corpus formal/peano-number-theory.mm \
  outputs/unified-mcts/corpus --output outputs/unified-mcts/audit.json

python -m neural_prover train outputs/unified-mcts/corpus outputs/unified-mcts/model \
  --device cpu --epochs 1 --batch-size 8 --d-model 64 --layers 2 \
  --candidate-loss-weight 0.35

python -m neural_prover build-benchmark formal/peano-number-theory.mm \
  outputs/unified-mcts/benchmark.json \
  --training-corpus outputs/unified-mcts/corpus \
  --reference-output outputs/unified-mcts/reference.private.json

python -m neural_prover evaluate-mcts outputs/unified-mcts/model/final.pt \
  outputs/unified-mcts/corpus outputs/unified-mcts/benchmark.json \
  formal/peano-number-theory.mm outputs/unified-mcts/evaluation.json \
  --device cpu --policies uniform,heuristic,neural,hybrid --seeds 17,19,23 \
  --simulations 40 --max-depth 16 --branching 24 \
  --external-verifier /path/to/metamath --require-external-verification
```

必须将 `/path/to/metamath` 换成实际程序路径。也可省略 `--external-verifier` 并设置
`METAMATH_EXECUTABLE`；不要省略正式评估所需的 `--require-external-verification`。
缺失、超时或验证失败不会计为认证成功。对照结果不得包含训练时的私有参考证明。
MCTS 的 `collect-replay` / `reinforce`、中间引理与潜在模型升级示例见
[路线与验收](progress-and-roadmap.md)和[潜在推理设计](lemma-and-latent-reasoning.md)。

## 5. HTPS：DAG 监督与同步闭环

以下显式使用小模型、CPU 和有限预算，验证新后端接口；不隐式启动约100M参数训练。
HTPS `init-model` 默认更大，后续规模实验需显式确定资源与配置。基础库示例不启用
PA+ 生成参数；启用方法见上述专题。模型上下文较小时可能跳过长样本，应检查数据和训练报告。

```bash
python -m htps_prover generate formal/peano.mm outputs/unified-htps/data \
  --steps 500 --seeds 7,11,19,23 --max-proof-depth 8

python -m htps_prover init-model outputs/unified-htps/data/tokenizer.json \
  outputs/unified-htps/initial.pt --d-model 64 --heads 4 \
  --encoder-layers 2 --decoder-layers 2 --ffn 512 \
  --max-state-tokens 384 --max-action-tokens 384 --thought-steps 2

python -m htps_prover train-supervised outputs/unified-htps/initial.pt \
  outputs/unified-htps/data/tokenizer.json outputs/unified-htps/data/policy_train.jsonl \
  outputs/unified-htps/sft.pt --lemmas outputs/unified-htps/data/lemma_train.jsonl \
  --device cpu --epochs 1 --batch-size 8 --max-examples 128

python -m htps_prover closed-loop outputs/unified-htps/sft.pt \
  outputs/unified-htps/data/tokenizer.json outputs/unified-htps/data/policy_train.jsonl \
  formal/peano.mm outputs/unified-htps/loop --device cpu \
  --iterations 2 --examples 8 --epochs 1 --simulations 40 --expansions 100 \
  --external-verifier /path/to/metamath --require-external-verification

python -m htps_prover evaluate outputs/unified-htps/loop/checkpoint-002.pt \
  outputs/unified-htps/data/tokenizer.json outputs/unified-htps/data/policy_test.jsonl \
  formal/peano.mm outputs/unified-htps/evaluation.json --device cpu \
  --limit 32 --simulations 40 --expansions 100 \
  --external-verifier /path/to/metamath --require-external-verification
```

若从旧模型启动，生成时传 `--base-tokenizer`，再用 `init-latent` 升级配套权重，
不随意重建 tokenizer。两个独立命令 `collect` / `train-replay` 对应同步闭环的
搜索与训练阶段。外部未通过的已解路径不写入成功 replay；预算耗尽不是伪负样本。
完整语义和未实现的跨节点 batch、异步 actor 等见 [HTPS 设计](htps-design.md)。

## 6. Scale 与公平比较

`build-scale-corpus` / `audit-scale-corpus` / `train-scale` 保留，支持模板实例化、
gzip 分片、长上下文和断点恢复。它们是规模化数据/训练工具，不是独立形式系统。
配置参考[历史百万级训练](first-large-scale-run.md)与
[RTX 5090 实验](first-rtx5090-closed-loop-run.md)，不要直接套作 PA+/HTPS 的规模结论。
2026-10-03 已接入 PA+ 桥模板生成及可选候选损失。
[RTX 3090 长时训练入口](pa-plus-rtx3090-session.md)提供时间预算、闭式实例重采样和本地监控。

候选评分头现在使用保留动作顺序的双向 GRU，检查点记录
`candidate_policy.encoder = bidirectional-gru-v1`。旧均值池化检查点仍可加载，
但其候选头会回退到自回归评分；只有重新训练新候选编码器后才应启用候选索引评分。
由于增加了参数，旧优化器不能用于新版的精确断点恢复，须从旧权重开始一次使用新优化器的微调。
已启动的训练进程继续使用其启动时的代码，不会在运行中自动切换架构。

Scale 的新检查点同时保存 `tokenizer_sha256`；证明、评估、回放采集和 Scale 续训入口
统一检查指纹、词表大小与特殊 token ID。历史上没有指纹的检查点只能检查后两项，
因此必须保留其原配 tokenizer，不能据此声称普通 token 的顺序已经得到验证。

中间引理采用局部 cut 作用域：先证明引理，再用于当前目标及其子目标，
结束后退出作用域才处理原有兄弟目标。完整回放保留激活和退出标记，沿用现有词表。
来自旧全局引理作用域的显式 cut 轨迹应重新生成、重放和认证。
MCTS 对长度过滤、有限候选枚举和分支预算产生的空候选保留“未知”，不写成不可证负例。

正式对比仍需补齐：

1. 相同目标族、训练信息、允许规则及验证要求，保留未参与35目标引导的目标族；
2. 生成引导、训练/推理模型标签和推理排序分别开关，核对环境指纹；
3. 不直接比较 MCTS 与 HTPS 的 simulation 数，记录实际展开、候选评分与墙钟时间；
4. 至少三种子，报告深度分层解题率、失败类型和额外潜在计算成本；
5. 仅将最终证书通过指定认证门槛计为成功。

现有 MCTS benchmark JSON 与 HTPS policy JSONL 不同，尚无统一跨后端评估调度器。
以上两组示例分别检查各自路径，不能直接拼成公平的 MCTS 对 HTPS 胜率结论。
