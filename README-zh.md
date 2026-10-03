[![English](https://img.shields.io/badge/Language-English-lightgrey)](README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue)](README-zh.md)

# Peano Metamath Prover

一个基于皮亚诺算术及其保守 PA+ 定义的合成定理生成与神经符号证明研究工具。
训练数据来自实际构造的形式证明，而非自然语言推理文本。神经模型提出候选或为
动作排序；符号内核检查动作，并重放最终的 Metamath 证明证书。

PA+、MCTS、HTPS、中间引理和连续潜在推理统一在 `main` 维护，无需切换算法专用
分支或单独安装 HTPS。本项目处于 Beta 研究阶段：工程管线已可运行，但尚未证明
学习策略能够提升实际定理解题能力。

独立目录 [v2](v2/README.md) 新增 RoPE decoder-only 主模型、因果引理编码器、
已认证定理库、显式草稿纸 Agent、标准下一 token 预训练与 RLVR。
新版本复用 PA+ 内核；不迁移 v1 检查点。

## 1. 系统组成

| 层次 | 实现位置 | 职责 |
| --- | --- | --- |
| 共享形式系统 | `formal/`、`metamath_generator` | PA / 数论 / PA+；解析、合一、类型、`$d` 和证明构造 |
| 合成数据 | `metamath_generator`、`htps_prover.forward` | 质量过滤的定理、有界实例、定义桥与证明 DAG 数据集 |
| 模型与训练 | `neural_prover`、`htps_prover.training` | 形式符号词表、策略/价值头、可选中间引理动作与连续潜在计算 |
| 搜索后端 | `neural_prover`、`htps_prover.hypergraph` | 多目标状态上的 best-first / PUCT-MCTS，或共享 AND/OR 目标节点上的 HTPS |
| 认证 | 共享环境与证书验证器 | 对源库重放；可选强制外部 Metamath 验证 |
| V2 证明 Agent | `v2/pa_prover_v2` | 因果语言模型与上下文引理向量；NTP/SFT/RLVR 和可验证草稿纸动作 |

MCTS 和 HTPS 共享数学语义，但并非仅有数值参数不同：搜索结构和回放目标也不同。
现有 CLI 与数据格式保留为可选工作流。PA+ 目录驱动生成已接入 `build-corpus`，
现也已贯通 HTPS `generate`、policy/lemma 动作审计与训练。
Scale 生成器也已接入由认证桥支持的 PA+ 模板重放。接线矩阵与命令见
[训练与评估指南](docs/training-and-evaluation.md)。

## 2. 形式系统与可信边界

形式库按层次组织：

- `formal/peano.mm`：基础皮亚诺算术库。
- `formal/peano-number-theory.mm`：保守数论定义与命名目标。
- `formal/peano-pa-plus.mm`：从 `formal/pa-plus-definitions.json` 生成的68个新增
  高层定义与35个闭合目标公式。

PA+ 覆盖算术关系、有限递推、整数/有理数及初等分析的有限证书，并不是通用实分析库。
`statement` 只命名公式，不宣称它成立：35个目标既不是新增公理，也不是已解定理清单。
目标引导采样不等于证明了目标。

定义编译器检查依赖、自由变量与新鲜性约束。136条展开/折叠桥是有证明支撑的搜索宏，
最终证书会将其内联回源库规则。中间引理必须先证后用；潜在向量不能创建事实。
详细说明见 [PA+ 定义](docs/pa-plus-definitions.md)与[系统架构](docs/architecture.md)。

项目内验证器目前支持未压缩证明。重要结果还应通过外部 Metamath 实现交叉验证。
神经评分、搜索节点标记 solved 或很高的 token accuracy 都不是证明。

## 3. 安装与运行

使用 Python 3.10+，在仓库根目录执行。仅使用符号生成器无需 PyTorch：

```bash
python -m pip install -e .
python -m metamath_generator formal/peano.mm \
  --mode random --steps 100 --seed 7 --output-dir outputs/smoke
```

神经训练与完整 Python 测试集：

```bash
python -m pip install -e ".[neural,dev]"
python -m pytest
```

本机已有 PyTorch 时，可直接从源码测试，无需重新安装：

```powershell
$env:PYTHONPATH='src;v2'
python -m pytest
```

Bash 示例用 `\` 续行，PowerShell 请改为反引号或合并为一行。真实外部验证集成测试
另需 Metamath 可执行文件；测试跳过不等于外部认证通过。上述测试不会启动 GPU 训练。
实验输出与检查点均被 Git 忽略。

尝试共享 PA+ 生成器，启用有界自然数实例和目标引导：

```bash
python -m metamath_generator formal/peano-pa-plus.mm \
  --mode random --steps 1000 --seed 7 \
  --definition-catalog formal/pa-plus-definitions.json \
  --bootstrap-definitions --definition-coverage-weight 3 \
  --bounded-nat-max 2 --ground-instances-per-predicate 1 \
  --target-guidance-weight 4 --max-definition-only-search-per-predicate 1 \
  --output-dir outputs/pa-plus-guided
```

生成器输出 `closed_theorems.jsonl`、`inference_rules.jsonl`、`proof_states.jsonl`
及质量报告。PA+ 语料审计/训练、MCTS 基线或 HTPS 监督/闭环实验，请继续阅读
[训练与评估指南](docs/training-and-evaluation.md)。

## 4. 评估原则与当前证据

主要指标是明确计算预算下的认证解题率，而不是动作 token 准确率。参考证明不得
进入搜索输入，构造 benchmark 时必须检查训练重叠。比较 uniform、heuristic、neural
和 hybrid 时，应固定目标、允许使用的规则、种子与预算，并报告证书状态和环境指纹。
MCTS 与 HTPS 使用相同 simulation 次数，并不意味着计算预算相同。

`neural_prover evaluate` / `evaluate-mcts` 和 HTPS `evaluate` / `collect` / `closed-loop`
支持 `--external-verifier /path/to/metamath --require-external-verification`。
要求外部验证时，程序缺失、失败、超时或结果不明确均不计为认证成功。
`internal_only` 与 `internal+external` 的结果必须分开统计。生成引导、模型目标提示与
推理排序有独立消融开关，见[认证与公平评估](docs/pa-plus-certification.md)。

截至 2026-09-04 的证据：

- 认证：重新生成的616例 PA+ 语料全量动作回放通过；三个定义桥证书和一个 HTPS
  sanity 目标通过外部重放。旧 PA+ 桥动作的变量槽位不稳定，必须用 `sorted-v1`
  重新生成；历史检查点予以保留，不静默修补。
- 历史 Scale/MCTS 实验：生成1.1M条记录，105M参数模型训练10,000步，实际消费
  80,000条记录。九个固定目标上 uniform/heuristic/neural/hybrid 分别解出
  0/9、4/9、0/9、3/9；七个成功 attempt 全部通过外部验证。这不是 PA+ 或 HTPS 训练。
- PA+ GPU 与 latent/HTPS 工作仍属于工程验证。尚无学习带来解题率提升的证据，
  没有35目标证明成功的结果，也没有完成大规模 PA+/HTPS 实验。

PA+ → HTPS 数据/训练及基础 checkpoint 微调已贯通。下一步建立保留目标族，完成
多种子受控评估，再考虑扩大训练。详细证据与验收标准见[进展与路线](docs/progress-and-roadmap.md)。

## 5. 项目结构与文档

```text
formal/                  共享 PA、数论、PA+ 形式库与定义目录
src/metamath_generator/  形式内核、保守定义与合成数据生成
src/neural_prover/       模型、MCTS、Scale、共享环境与认证
src/htps_prover/         前向 DAG 数据、HTPS、latent 训练与同步回放
tests/                   共享内核及两个后端的回归测试
tools/                   验证工具
docs/                    使用指南、设计、审计与历史实验报告
outputs/                 本地数据/检查点/报告（不跟踪）
```

[文档导航](docs/README.md)按任务而非分支组织资料。目前详细设计文档主要为中文。

- 运行实验：[训练与评估](docs/training-and-evaluation.md)。
- 理解实现：[系统架构](docs/architecture.md)、[HTPS](docs/htps-design.md)、
  [中间引理与潜在推理](docs/lemma-and-latent-reasoning.md)。
- 开发 PA+：[定义](docs/pa-plus-definitions.md)、[生成](docs/pa-plus-random-generation.md)、
  [神经管线](docs/pa-plus-neural-training.md)。
- 查看结果与规划：[路线](docs/progress-and-roadmap.md)、
  [形式系统演进](docs/formal-system-evolution.md)。

历史 `htps` 分支保留，但新增开发统一进入 `main` 或短期功能分支。既有命令保持可用；
旧 `peano-metamath-prover-htps` 安装名的迁移方法见训练指南。
贡献约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 6. 许可证

[GPL-3.0](LICENSE)。`formal/peano.mm` 保留 Robert Solovay 的原始版权声明，
详见 [NOTICE](NOTICE)。
