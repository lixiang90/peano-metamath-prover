# 当前进展与研究路线

本文记录截至 2026-08-31 项目已经实际完成的能力、尚未证明的能力，以及下一阶段
闭环解题、强化学习和连续潜在思维的实现顺序。规划中的功能不会描述成现有能力。

关于 PA+/Metamath 定义层、Equations+、Lean 无自然语言合成数据和多后端迁移实验
的后续方案，见[形式系统演进设计](formal-system-evolution.md)。

## 0. P0–P3 实施状态

这里把“工程实现”和“研究验收”分开。代码能运行不等于实验假设已经成立。

| 阶段 | 工程状态 | 研究验收状态 |
| --- | --- | --- |
| P0 可信闭环 | 已实现 benchmark 泄漏审计、逐题规则排除、四策略同预算 MCTS、项目内/外部双验证、Wilson 95% 区间、计算计数和逐 attempt 进度日志 | 已完成 9 题单 seed 首轮；7/7 成功证书通过官方 Metamath；仍待 37 题三种子全量报告 |
| P1 结构化策略 | 已实现对符号内核枚举的有限候选集合进行候选索引打分；替换仍由合一器求解；旧检查点保留自回归冷启动兼容路径 | 首轮使纯神经搜索加速 2.32 倍，但 solve rate 未提升，研究验收未通过 |
| P2 可验证 RL | 已实现 AlphaZero 式 MCTS 访问分布/价值回放、失败死端价值样本、预算耗尽删失和候选策略头更新 | 50 条 replay 首轮 solve rate 不变，研究验收未通过；GRPO/PPO 对照尚未实现 |
| P3 分解与潜在思维 | 已实现受内核约束的中间引理 cut 动作、词表兼容升级、连续向量递归、停止头、MCTS/replay 接线与证书内联 | 仅完成本地工程原型与回归测试；尚未规模训练或证明 solve rate 增益 |

P3 已按新需求提前实现为隔离的可选原型，旧模型文件与检查点不变。它不会绕过
P0–P2 的研究门槛：新增权重训练和规模对照实验应在现有可信闭环上进行。

### 2026-08-24 RTX 5090 首轮正式规模结果

- 生成 1,100,000 条动作记录，10,000 条确定性抽样全部通过内核回放；
- 在 RTX 5090 上将 105,312,496 参数模型训练至 10,000 步，消费 80,000 个
  样本和 73,038,051 个目标 token；
- 5,000 条测试的 token accuracy 为 92.26%，完整参考动作准确率仍为 0；
- 从 37 个泄漏合格 research 目标中固定抽 easy/medium/hard 各三题，seed=17、
  40 simulations，四策略共 36 attempts；
- SFT 与候选头更新后均为 uniform 0/9、heuristic 4/9、neural 0/9、hybrid 3/9；
- 12 个训练目标收集到 50 条 replay；候选头更新使 neural 总时延
  147.38→63.62 秒，但没有提高 solve rate；
- 最终 7 个成功 attempt 全部通过项目内核和官方 Metamath 双验证。

完整配置、哈希、逐策略耗时和限制见
[首次 RTX 5090 规模训练与闭环实验](first-rtx5090-closed-loop-run.md)。该结果是
首轮受控实验，不替代 37 题、seeds 17/19/23 的 P0 正式验收。

## 1. 已完成的基础

### 形式系统

- 解析并区分语法规则、`|-` 逻辑断言和只命名开放目标的 `statement`；
- 支持一阶合一、类型检查、occurs check、`$d` 约束和变量 standardize-apart；
- 支持部分前提消解，将证明统一表示为 `Γ ⊢ C`；
- 保存证明 DAG、证明内部变量闭包，并导出未压缩 Metamath 证明；
- 验证器按声明顺序和活动作用域重放，拒绝未来标签和失效局部假设；
- 使用质量过滤、语义去重、前提 subsumption 和 proof-state 配额控制低质量爆炸。
- 生成式 PA+ 目录已增加 68 个无环保守定义和 35 个闭合目标公式，覆盖数论、
  有限递推、整数/有理数及初等分析证书；定义编译器机械审计自由变量、依赖顺序、
  `$d` 新鲜性和 `statement` 非逻辑边界。
- PA+ 随机生成器已能从 `df-*` 机械导出可重放的展开/折叠桥，按目录报告覆盖率，
  使用规范小自然数项生成类型安全闭式实例，并用35个非逻辑 `statement` 的子公式
  结构做反向候选引导；定义依赖溯源配额会限制单一定义包装。固定 `seed=7` 的
  1,000步对照从34/35提高到35/35目标触达，包装配额拒绝111个候选且保持35/35；
  仍待真正的开放子目标反向搜索和定义完全展开后的语义新颖性过滤。

### 数据和训练

- 正式 Scale 语料包含 1,000,000 条无重复动作记录；
- 训练/验证/测试为 990,000/5,000/5,000，训练集覆盖证明深度 1–16；
- 10,000 条随机样本全部通过项目内符号环境回放；
- 104,130,543 参数 encoder-decoder Transformer 在 RTX 3090 上训练到 5,000 步；
- 该阶段消费 40,000 样本、约 3,099 万目标 token，并保存可精确续训检查点；
- 自然测试集 token accuracy 为 88.31%，深度压力测试总体为 84.78%；
- 符号生成压力测试达到证明深度 19，神经评估语料覆盖到深度 17。

详细配置和逐深度结果见
[首次百万级训练与深度实验](first-large-scale-run.md)。大型语料和检查点位于
Git 忽略的 `outputs/scale-gpt2-depth-v1/`，不随源码仓库提交。

## 2. 当前模型究竟学了什么

当前监督训练输入一个证明状态，目标是复现语料记录的下一步动作：

```text
state = 当前目标 + 开放前提
target = 参考规则 + 父对象 + 显式替换
```

损失由动作 token 交叉熵和辅助价值误差组成。这适合冷启动策略，但存在明确
边界：

- 它奖励复现某一条参考轨迹，而合法证明可能不唯一；
- token accuracy 高不保证整条动作合法；
- exact-reference-action 为零不等于完全解不出题；
- 单步动作评估没有测量多步规划和回溯。

因此这些指标以后只作为优化诊断。项目的主要研究指标必须转为实际闭环解题。

## 3. 闭环解题与正确评估

### 目标构造

对每个 benchmark 目标，只向证明器提供前提和结论。参考证明只用于确定难度和
参考深度，不能进入模型输入或搜索数据库。还需要：

- 移除目标定理及其 alpha-equivalent、同一规范化结论副本；
- 按规范化结论和证明 DAG/规则骨架分组切分，降低模板泄漏；
- 保留基础公理、语法规则和明确允许使用的先前定理；
- 分别构造深度 1–4、5–8、9–12、13–16、17+ 的目标集。

### 闭环执行

```text
ProofState
    ↓
符号候选生成器：枚举类型与 $d 基本合法的动作
    ↓
策略模型：对规则、父定理和前提消解候选排序
    ↓
Beam / PUCT-MCTS：选择、展开、回溯
    ↓
BackwardEnvironment：执行合一和真实状态转移
    ↓
全部开放前提消解？──否──→ 继续搜索
    │是
    ↓
编译证明 DAG → 项目内验证 → 外部 Metamath 交叉验证
```

### 主要指标

- `certified_solve_rate@nodes` 和 `certified_solve_rate@seconds`；
- 按参考深度、目标结构和开放前提数分层的解题率；
- 找到证明所需节点数、时间、证明步骤数和最大深度；
- 合法候选率、证书编译率和外部验证通过率；
- 至少三个随机种子的均值、置信区间和失败类型。

必须并列报告四条基线：纯符号、手工启发式、神经策略、神经+符号 MCTS。
只有神经策略相对纯符号基线的增益才能归因于学习。

## 4. 结构化动作优先于自由文本动作

当前动作可能包含很长的显式替换序列，单个 token 错误就使整条动作失败。下一版
优先把网络输出改成有限结构：

```text
rule_id
parent_theorem_ids
premise_matching
open_premise_mask
optional macro/lemma proposal
```

替换尽可能由符号合一器自动求解，类型与 `$d` 检查始终由内核完成。网络输出
候选分数而不是自行拼写整个 Metamath 动作。这与 AlphaGeometry 的基本分工
一致：神经部分提出高价值选择，传统推理器完成精确、可验证的推导。

## 5. 强化学习：采用 R1-Zero 原则，不照搬起点

DeepSeek-R1-Zero 的可迁移核心是 RL with verifiable rewards：不要求复现人工
思维链，只根据最终可验证结果优化策略。Metamath 具有确定性验证器，尤其适合
这种奖励。计划对同一个目标采样一组搜索轨迹，并使用组内相对优势进行 GRPO，
或用 MCTS 访问分布进行 AlphaZero 式策略改进。

不计划从随机策略执行“纯 Zero”训练。当前模型只有约 104M 参数，长证明的终局
奖励很稀疏；如果同组轨迹全部失败，GRPO 几乎没有学习信号。合理顺序是：

1. 用当前监督模型冷启动候选排序；
2. 从浅层且纯符号基线可解的目标开始课程训练；
3. 通过 MCTS 产生多条真实搜索轨迹；
4. 仅将验证器接受的闭合证明视为成功；
5. 随 solve rate 提升逐渐增加证明深度和搜索预算。

终局奖励建议以证书为中心：

```text
+1.0  完整证明且证书验证通过
 0.0  搜索预算耗尽
负值  非法动作或破坏形式状态
```

可以加入很小的节点、思考和成功证明长度成本，但不能奖励与参考动作相似、输出
更长、表面深度更大或未经验证的“合理性”。若使用中间奖励，应优先采用基于
状态势函数的 shaping，避免改变最优策略。

GRPO 不需要单独 critic，但项目已有价值头且环境适合树搜索，因此必须比较：

- MCTS + value/policy 的 AlphaZero 式更新；
- GRPO 的组内终局奖励更新；
- PPO/actor-critic；
- 仅监督学习的固定策略。

DeepSeek-R1 论文：<https://arxiv.org/abs/2501.12948>。

## 6. 连续潜在思维与停止头

连续向量思维不是 R1-Zero 的组成部分，更接近 Coconut。拟议结构只在正式动作
之前执行内部计算：

```text
h0 = Encode(ProofState)
h1 = Think(h0)
...
hk = Think(h{k-1})
stop_head(hk) → 输出结构化动作
```

潜在思维可以帮助比较多个候选、估计后续分支或规划辅助引理，但不能直接改变
形式状态。每次真实规则应用仍必须：

1. 输出离散结构化动作；
2. 由符号环境执行；
3. 记录到证明 DAG；
4. 最终进入可回放证书。

停止头需要防止“立即停止”和“永不停止”两种退化。实验应设置最大思考步数，
比较固定 0/2/4/8 步，再引入可学习 `continue/act` 决策和很小的 ponder cost。
成功奖励必须远大于计算成本。

训练顺序建议：

1. 固定思考步数的监督消融；
2. 用 MCTS 搜索改进量或候选不确定性训练思考表示；
3. 学习停止头；
4. 最后联合 RL，比较相同搜索预算下的 solve rate。

Coconut 论文：<https://arxiv.org/abs/2412.06769>。

## 7. 分阶段路线与验收标准

### P0：可信闭环评估

- 实现不会泄漏参考证明的 benchmark 构造；
- 输出按深度分层的 certified solve rate；
- 对成功样本生成证书并执行项目内、外部双验证；
- 建立纯符号、启发式、神经和混合 MCTS 四组基线。

验收：报告能回答“在相同预算下，每种策略实际证明了多少目标”，而不是只报告
动作 token accuracy。

当前实现说明：公开 benchmark 不再携带 `reference_rule`；需要参考深度/规则时写入
私有 sidecar。构建器会索引训练语料中的规范化证明状态，剔除精确或
alpha-规范化结论碰撞，并为每题排除源库中同结论标签。评估默认只统计
`research` 组；`foundational` 只用于 sanity，著名定理与开放猜想属于 frontier。
每个策略、题目和 seed 都使用全新的环境实例，避免候选缓存给后运行策略带来优势。
成功证书可以要求外部 Metamath 验证器通过，否则按 fail-closed 处理。

### P1：结构化策略与高效搜索

- 网络排序符号合法候选；
- substitution 交给合一器；
- 增加候选缓存、批量评分、长度分桶和动态 token batch；
- 报告节点吞吐、GPU 利用率与 solve-rate/compute 曲线。

验收：神经策略在至少一个非平凡深度桶中显著优于纯符号基线。

当前实现说明：策略头接收一个状态和内核已经枚举、类型检查过的有限动作集合，
输出候选索引 logits；不生成 substitution 文本。MCTS replay 第一次提供有效访问
分布后，检查点会标记 `candidate_policy.trained=true`，推理随即切换到候选索引头。
旧监督检查点没有这些权重时仍可加载，并以原自回归动作似然作为 bootstrap 排序。
环境和策略报告枚举次数、缓存命中、状态转移次数、候选数和评分 token 数。

### P2：可验证强化学习

- 收集完整 MCTS 轨迹和验证结果；
- 实现 AlphaZero 式更新与 GRPO 对照；
- 使用浅到深课程、失败回放和多随机种子；
- 监控奖励投机、证明膨胀和搜索长度偏置。

验收：固定搜索预算下 certified solve rate 提升，并保持证书 100% 内核通过。

当前实现说明：成功路径同时提供策略和价值监督；已证明无动作的死端只提供负价值
监督；搜索预算耗尽既不是失败证明，也不产生伪负价值，按删失样本记账。策略损失
是同一状态有限合法候选上的交叉熵，目标为 MCTS 访问分布。该实现是 P2 的
AlphaZero 路线；GRPO/PPO 只有在形成稳定的同预算基线后才进入对照实验。

### P0–P2 最小复现实验

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

python -m neural_prover collect-replay \
  outputs/model/final.pt outputs/corpus formal/peano-number-theory.mm \
  outputs/replay.jsonl

python -m neural_prover reinforce \
  outputs/model/final.pt outputs/replay.jsonl outputs/model/rl-1.pt
```

正式结论必须比较 RL 前后完全相同的题目选择、seed、simulation、branching、
最大深度和外部验证要求。报告中的 attempt 是“题目 × seed”，不能把同一道题的
多个 seed 当作互相独立的新定理。

### 2026-08-22 端到端 smoke 结果

本轮用小模型和很小预算验证 P0–P2 的接线，而不是做研究验收：

- 生成 71/1/4 条 train/validation/test 样本，benchmark 索引了全部 71 个训练状态；
- 公开 benchmark 有 9 个 research-eligible 目标，且不含 `reference_rule`；
- 从每个 easy/medium/hard 桶固定抽 1 题、seed=17、20 simulations，四策略共
  12 次 attempt，5 次通过项目内核和官方 Metamath 双验证；
- 冷启动分组为 uniform 0/3、heuristic 2/3、neural 1/3、hybrid 2/3；
- 6 个训练目标全部产生可认证证明，形成 13 个策略/价值均已知的去重 replay 状态；
- 一轮 AlphaZero 更新后，检查点推理报告 `scoring_mode=candidate_index`；
- 同题同预算复跑 neural/hybrid 得到 0/3 和 2/3，合计从更新前 3/6 变为 2/6。

最后一项明确说明这个 smoke **没有**证明 RL 改善，反而在极小样本上退化。它只
证明候选头训练、检查点切换、闭环搜索和双验证可以贯通。P1/P2 的研究验收必须
扩大 replay、加入保留评估集和三种子置信区间；P3 原型在此之前不应被描述为能力
提升，也不应成为扩大模型的依据。

### P3：连续潜在思维

- 工程原型已完成：中间引理动作、固定预算连续递归、可学习停止头和最大思考预算；
- 用认证 replay 训练新增词表行、潜在递归层、停止头和引理门控头；
- 固定 0/2/4/8 思考步数消融；
- 比较无思维、离散辅助线提议和连续思维；
- 报告性能、额外计算量和按深度收益。

验收：在相同或明确计价的计算预算下，提高深层目标解题率，而不降低证书可信度。

### P4：扩展与发布

- 更紧凑的状态/DAG 表示或更长上下文，覆盖深度 18+；
- 多进程符号生成和长度/难度分桶训练；
- 外部 Metamath 验证率、多个 seed 和置信区间；
- 再决定是否续训现有检查点到 10,000 步及扩大模型。

只有 P0–P2 建立真实解题增益后，扩大数据与模型才具有明确研究意义。
