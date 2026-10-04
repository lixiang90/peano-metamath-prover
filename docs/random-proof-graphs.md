# 随机证明图生成

`graph` 是定理生成、v1 语料、HTPS 语料、Scale 和 v2 的新默认算法。
它借鉴 HTPS 附录 E.5 的前向证明图构造，由本项目的 Metamath 组合和验证内核实现，
没有复制 Evariste 的代码，也不引入新的公理或外部数学证明数据。

## 算法

1. 用加载的逻辑公理/定理初始化节点库；`statement` 目标不是可用事实。
2. 选择一个含前提的源规则，或以前生成的含前提定理。
3. 先匹配结构较具体的前提，再匹配较自由的前提；在同一个合一过程中求解替换。
   所有规则和父节点变量先分离命名空间，检查 occurs check、类型及 `$d` 条件。
4. 已匹配父节点的假设与未匹配前提一起成为新定理的假设；不会把假设当成闭合事实。
5. 新结论与完整依赖入库，之后可以作前提，也可以作规则。重复声明按 alpha 标准化去重。

默认 **90% 尝试执行组合，10% 尝试执行小表达式实例化**。实例化对象可以是已有生成
定理或源公理；它是辅助多样性操作。这个概率是尝试比例，经过重复、类型和资源过滤后，
产出的比例不一定正好为 90:10。设为 0 后仍可通过合一产生不同证明结构。

组合默认以 25% 的概率尝试使用已有派生规则。规则按使用次数进行稀有性采样，父节点按
深度分桶采样，再降低重复使用节点的权重。每个前提最多检查 24 个候选；无匹配时保留
开放前提。没有固定证明模板或人为的“深度超过 6 扣分”准入机制。

质量分析仍记录分数，但不会从工作图中删除简单中间结论。AST 大小、变量数、假设数、
证明依赖深度和生成尝试数仍有明确预算；无效或超预算样本不会作为失败证明训练。

## 证书与数据

- 派生规则本身也记录为 DAG 依赖，导出顺序保证引用在先；不会把派生规则转成 `$a`。
- 一张图只导出、解析一次，独立重放全部生成的 `$p`，再把派生宏展开回源规则。
- v1 训练提取每条完整证明中实际实例化的全部逻辑步骤，并去重，使用原始假设与 `$d`。
  所有序列化动作再次通过反向环境检查。源规则的语法证明不会变成逻辑训练动作。
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
python -m metamath_generator formal/peano-pa-plus.mm --mode graph --steps 2000 --max-proof-depth 12 --instance-probability 0 --output-dir outputs/graph-raw

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
Scale 的图模式按有界批次串行处理；`workers` 目前仅用于旧模板模式。分片使用哈希验证，
保存图编号和图内游标；恢复时校验形式系统、词表和生成配置。`max_graph_batches` 限制
尝试批次，耗尽会保存未完成状态并报错，不会用重复数据填满配额。

旧算法可显式选择：原始 CLI `--mode random`，v1/HTPS/v2 `--generation-mode random`，
Scale `--generation-mode templates`。已有数据和已训练检查点不会自动变成新算法产物；
请使用新的输出目录重新生成数据。v2 的固定公理 warmup 和定义桥默认关闭，仍可显式启用。

参考：[HTPS 主论文](https://papers.neurips.cc/paper_files/paper/2022/file/a8901c5e85fb8e1823bbf0f755053672-Paper-Conference.pdf)，
[附录 E.5](https://papers.neurips.cc/paper_files/paper/2022/file/a8901c5e85fb8e1823bbf0f755053672-Supplemental-Conference.pdf#page=12)。

## 2026-10-04 验证

- 完整回归：257 项测试通过，包括无辅助实例化时的深图生成、派生规则的拓扑依赖、
  跨 `PYTHONHASHSEED` 复现、分片恢复与连续生成一致、全部序列化动作回放。
- PA+：种子 7、11，各 250 次图生成尝试；2,710 条去重后的逻辑步骤，展开深度最大 14。
  两份完整图导出文件中的证明全部通过官方 C Metamath 验证。
- v2：种子 7、11，各 100 次尝试、证书配额 32；最终得到 31 条完整 Agent 轨迹，
  全部证书通过官方 Metamath。另有一个候选超出证书标签预算，一个轨迹超出上下文预算，
  均显式记录为过滤项，没有当作错误证明训练。
- HTPS：种子 19、60 次尝试；236 条策略步骤、191 条引理提议，427 条动作全部重放通过。

这些结果验证数据和证明的可靠性，不表示重新训练后的模型已经获得更强的证明能力。
