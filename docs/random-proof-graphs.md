# 随机证明图生成

`graph` 是定理生成、v1 语料、HTPS 语料、Scale 和 v2 的新默认算法。
它借鉴 HTPS 附录 E.5 的前向证明图构造，由本项目的 Metamath 组合和验证内核实现，
没有复制 Evariste 的代码，也不引入新的公理或外部数学证明数据。

## 算法

1. 用加载的逻辑公理/定理初始化节点库；`statement` 目标不是可用事实。
2. 选择一个含前提的源规则，或以前生成的含前提定理。
3. 对至少有两个前提的规则，默认以 30% 的概率均匀选一个前提主动保留，
   其余 70% 尝试匹配全部前提。对待匹配的前提，先处理结构较具体的，再处理较自由的；
   在同一个合一过程中求解替换。
   所有规则和父节点变量先分离命名空间，检查 occurs check、类型及 `$d` 条件。
4. 已匹配父节点的假设与未匹配前提一起成为新定理的假设；不会把假设当成闭合事实。
5. 新结论与完整依赖入库，之后可以作前提，也可以作规则。重复声明按 alpha 标准化去重。

默认 **90% 尝试执行组合，10% 尝试执行小表达式实例化**。实例化对象可以是已有生成
定理或源公理；它是辅助多样性操作。这个概率是尝试比例，经过重复、类型和资源过滤后，
产出的比例不一定正好为 90:10。设为 0 后仍可通过合一产生不同证明结构。

组合默认以 25% 的概率尝试使用已有派生规则。规则按使用次数进行稀有性采样，父节点按
深度分桶采样，再降低重复使用节点的权重。每个前提最多检查 24 个候选；无匹配时保留
开放前提。主动保留与匹配失败是两种独立原因：例如规则 `A(x), B(x) => C(x)`，
若旧节点通过合一提供 `A(t)`，而 `B(x)` 被主动保留，就产生 `B(t) => C(t)`；
父节点自身的假设也要一并继承。合一后重复的假设只保留一份。
单前提规则不主动留空，每次组合至少接入一个旧节点，避免只复制原规则。
没有固定证明模板或人为的“深度超过 6 扣分”准入机制。

结构预筛选同时识别叶变量和算子变量（如 `QUANT`、`BINPRED`、`BINOP`、`LOGBINOP`）。
算子变量匹配仍检查参数个数和子结构；叶变量可代表整棵表达式。不同类型名不直接视为
不兼容，因为 PA 的 `var` 表达式也可作为 `term`；实际类型、occurs check 和 `$d` 条件
由后续组合内核检查。

`graph_partial_premise_probability=0.30` 是多前提组合尝试的采样概率，不是最终训练
数据的硬配额，也不是测得的最优值。主动保留一个前提可以提供开放推理规则，同时避免
一次引入太多假设。设为 0 关闭主动保留（匹配失败仍保留前提），设为 1 则所有多前提
组合都保留一个。假设数仍受 `max_hypotheses` 限制。

原始生成的 `quality_summary.json` 和各语料 manifest 中的图统计包含：

| 字段 | 含义 |
| --- | --- |
| `graph_multi_premise_plans` / `graph_partial_plans` | 可采样的多前提尝试数 / 主动保留的尝试数 |
| `graph_partial_plan_admitted` | 主动保留后通过检查和去重、实际入库的组合数 |
| `graph_nodes_with_new_hypotheses` | 合一后产生至少一个非继承假设的组合数 |
| `graph_nodes_with_reserved_new_hypothesis` | 主动保留的前提实际成为非继承假设的组合数 |
| `graph_new_hypotheses` / `graph_inherited_hypotheses` | 入库组合中新假设 / 父节点继承假设的去重计数之和 |
| `graph_unmatched_after_search` | 入库组合中尝试匹配但没有成功的前提槽位数，不含主动保留的槽位 |

这些入库统计以 `random_graph` 为分母，不包含辅助实例化；从完整证明提取训练步骤后，
分布还会变化。父节点继承的假设不误计为本次新增，同一公式与继承假设重合时也不重复计数。

质量分析仍记录分数，但不会从工作图中删除简单中间结论。AST 大小、变量数、假设数、
证明依赖深度和生成尝试数仍有明确预算；无效或超预算样本不会作为失败证明训练。

## 证书与数据

- 派生规则本身也记录为 DAG 依赖，导出顺序保证引用在先；不会把派生规则转成 `$a`。
- 一张图只导出、解析一次，独立重放全部生成的 `$p`，再把派生宏展开回源规则。
- 每次展开派生引理时，其非必需的证明局部变量独立改名；从源规则重建局部 `$d` 条件，
  不加强原声明变量之间的约束。展开后的证书再次独立验证，禁止跨调用泄漏旧 `$f` 标签。
- v1 训练提取每条完整证明中实际实例化的全部逻辑步骤，并去重，使用原始假设与 `$d`。
  所有序列化动作再次通过反向环境检查。源规则的语法证明不会变成逻辑训练动作。
  当前 v1 状态格式不能声明只在动作中出现的局部变量，这些步骤计入
  `graph_action_only_variables` 后跳过；不会丢弃证明图中的定理，也不会添加虚假假设来编码它。
- HTPS 使用同一批展开步骤生成策略样本，并从实际子目标产生通过内核检查的引理提议。
- v2 继续保留完整证书和经过重放的 Agent 轨迹；定理库仍仅使用训练分组，读取不能直接
  将声明变成事实。引理匹配改为有界的目标/已知前提联合合一，避免变量组合枚举爆炸。
- v1 新数据的 `value_target=1` 表示存在验证过的证明，是正例监督标签，不是实测搜索
  成功概率。搜索 critic 和 RLVR 的训练仍需实际搜索反馈；本改动没有启动新一轮训练。

v1 的 `proof_depth` 是展开后的源逻辑步骤深度；生成运行中的深度直方图是存储 DAG 的
依赖深度；v2 同时记录 `proof_depth` 与 `expanded_proof_depth`。二者都不是最短证明深度。
重复使用引理会压缩存储 DAG，因此展开深度可能超过 DAG 的配置上限，展开证书另有标签预算。

v1/Scale/新 HTPS 按 alpha 标准化结论族划分数据，避免同一结论的假设变体跨集合。
这不是“整个证明骨架未见过”的评测保证。v2 保留声明及完整逻辑证明骨架的连通分组。
完整根证明的泛化能力必须用独立证明评测衡量，不能用单步验证集准确率代替。

## 使用

在项目根目录安装后运行；PowerShell 可设置 `$env:PYTHONPATH='src;v2'`。

```powershell
# 原始随机图；完全关闭辅助实例化也可以生成新定理
python -m metamath_generator formal/peano-pa-plus.mm --mode graph --steps 2000 --max-proof-depth 12 --instance-probability 0 --partial-premise-probability 0.3 --output-dir outputs/graph-raw

# v1：全证明的源规则步骤；proof-graphs 下保存可独立检查的完整 .mm 文件
python -m neural_prover build-corpus formal/peano-pa-plus.mm outputs/graph-v1 --steps-per-seed 1000 --max-proof-depth 12 --instance-probability 0.1 --max-state-tokens 1024 --max-action-tokens 1024

# Scale：仅复用基础语料的 tokenizer，每批重新生成证明图，不读取证明模板
python -m neural_prover build-scale-corpus formal/peano-pa-plus.mm outputs/graph-v1 outputs/graph-scale --generation-mode graph --graph-steps 200 --max-proof-depth 12 --train-examples 10000 --validation-examples 1000 --test-examples 1000 --max-new-records 2000
# 重复同一命令可续生成；将 --max-new-records 设为 0 可完成全部目标数量

# HTPS policy / lemma 数据
python -m htps_prover generate formal/peano-pa-plus.mm outputs/graph-htps --steps 1000 --instance-probability 0.1

# v2 完整证明 Agent 数据
python -m pa_prover_v2 generate formal/peano-pa-plus.mm outputs/graph-v2 --steps-per-seed 300 --max-examples 120 --max-depth 12 --instance-probability 0.1
python -m pa_prover_v2 audit formal/peano-pa-plus.mm outputs/graph-v2 --require-external
```

外部验证前设置 `$env:METAMATH_EXECUTABLE` 为官方 `metamath.exe` 路径。
全部五个生成入口均支持 `--partial-premise-probability`，默认 0.3。
Scale 的图模式按有界批次串行处理；`workers` 目前仅用于旧模板模式。分片使用哈希验证，
保存图编号和图内游标；恢复时校验形式系统、词表和生成配置。`max_graph_batches` 限制
尝试批次，耗尽会保存未完成状态并报错，不会用重复数据填满配额。
本次新增采样参数会改变随机序列；旧图语料的 manifest 缺少该配置，不能直接续生成，
应使用新目录。旧模板模式的恢复配置不受新增图参数影响。

旧算法可显式选择：原始 CLI `--mode random`，v1/HTPS/v2 `--generation-mode random`，
Scale `--generation-mode templates`。已有数据和已训练检查点不会自动变成新算法产物；
请使用新的输出目录重新生成数据。v2 的固定公理 warmup 和定义桥默认关闭，仍可显式启用。

参考：[HTPS 主论文](https://papers.neurips.cc/paper_files/paper/2022/file/a8901c5e85fb8e1823bbf0f755053672-Paper-Conference.pdf)，
[附录 E.5](https://papers.neurips.cc/paper_files/paper/2022/file/a8901c5e85fb8e1823bbf0f755053672-Supplemental-Conference.pdf#page=12)。

## 2026-10-04 初版验证（主动保留前提改动前）

- 完整回归：257 项测试通过，包括无辅助实例化时的深图生成、派生规则的拓扑依赖、
  跨 `PYTHONHASHSEED` 复现、分片恢复与连续生成一致、全部序列化动作回放。
- PA+：种子 7、11，各 250 次图生成尝试；2,710 条去重后的逻辑步骤，展开深度最大 14。
  两份完整图导出文件中的证明全部通过官方 C Metamath 验证。
- v2：种子 7、11，各 100 次尝试、证书配额 32；最终得到 31 条完整 Agent 轨迹，
  全部证书通过官方 Metamath。另有一个候选超出证书标签预算，一个轨迹超出上下文预算，
  均显式记录为过滤项，没有当作错误证明训练。
- HTPS：种子 19、60 次尝试；236 条策略步骤、191 条引理提议，427 条动作全部重放通过。

这些结果验证数据和证明的可靠性，不表示重新训练后的模型已经获得更强的证明能力。

## 2026-10-04 主动保留前提验证

PA+ 使用种子 7、11、19，各 250 次尝试，DAG 深度上限 12、辅助实例化概率 0.1：

| 统计 | 主动保留概率 0 | 主动保留概率 0.3 |
| --- | ---: | ---: |
| 多前提组合尝试 | 208 | 238 |
| 主动保留尝试 | 0 | 67（28.2%） |
| 入库组合 | 415 | 442 |
| 主动保留且实际新增假设的组合 | 0 | 56 |
| 含新假设的组合（含匹配失败） | 154（37.1%） | 168（38.0%） |

改变采样会改变后续图和可选节点，这是一组小规模对照，不是证明能力提升的实验证据。

- 完整回归 267 项通过；覆盖主动保留比例、全部前提可匹配、单前提规则、继承假设去重、
  失败匹配、局部变量展开、跨哈希种子复现，以及 Scale 断点恢复。
- 六份完整图、共 957 个生成定理全部通过官方 C Metamath 验证。
- 另取两份含引理局部变量的展开证书（分别新增 6、1 个局部变量）通过官方验证。
- v1 提取 1,311 条训练步骤，序列化动作重放零错误；6 个动作局部变量步骤明确过滤。
- v2 生成 24 条完整 Agent 轨迹，24 份证书全部通过官方验证，无过滤项。
- HTPS 的 282 条策略步骤和 232 条引理提议全部通过动作重放。

## 2026-10-10 结构预筛选修复验证

- 全项目 280 项回归测试通过。
- 新增 13 项回归测试，覆盖四种算子变量的双向匹配、算子变量之间的匹配、整式变量、
  参数数量和子结构冲突、`var` 到 `term` 的类型兼容，以及组合内核拒绝非法类型替换。
- 在 PA+ 中添加已证明的恒等推理规则作为测试夹具，验证候选可进入 graph 和旧 random
  的候选池，并能导出、重放完整证明；测试不新增公理。
- 种子 7、11、19 各生成 250 步，三份完整图共 511 个生成定理全部通过官方 Metamath；
  每份图另验证一个最大 DAG 深度节点的展开证书，全部通过。
- 修复会改变候选集合及相同种子的生成序列。已有语料仍可读取；继续生成请使用新目录，
  避免沿用旧序列的图内恢复游标。
