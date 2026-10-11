# 完整状态监督、候选训练与 RLVR

2026-10-11。本文描述 v1 `neural_prover` 和 Scale 的新训练流程。前向随机证明图的
合一算法保持原有语义；变化在证明展开后的训练数据、训练抽样和搜索反馈。
HTPS 保留单个 AND/OR 节点监督，v2 保留其独立的 Agent/RLVR 管线。

## 1. 从完整证明重放待证目标队列

以前把证明的逻辑步骤拆开，每个样本只有一个目标，遗漏了搜索时其余尚未完成的义务。
现在从已经验证的完整证书恢复逻辑依赖，按反向环境的实际目标顺序重放。例如
`R → [P,Q] → [Q] → []`，中间状态包含两个目标。类型和语法组装不会变成逻辑动作；
`$f`、`$e` 交错顺序、局部假设及 `$d` 条件继续由内核检查。

一条轨迹必须全部重放到目标队列为空，并且每个状态/动作都能序列化后重新执行，才会
返回训练样本。超长或无法编码的轨迹整条跳过，统计为 `graph_trajectory_encoding`。
随后按状态/动作去重和划分集合，因此磁盘记录不保证包含每条轨迹的全部步骤。

默认先尝试无逻辑前提源规则的一步闭合，并复用同一证书中的已有证明计划；只有完整
闭合且不长于原轨迹才接受。这样复杂的 `t=t` 不会仅因原证书绕远而标成困难问题。
这是有预算的已知证明缩短，**不保证找到了最短证明**，也不等于实测搜索成功率。

新记录的 `proof_depth` 和 `remaining_steps` 是当前状态完成全部目标所需的已知剩余
反向动作数，`reference_steps` 是缩短前整条轨迹的动作数。它们不再是生成 DAG 深度。
`trajectory_id`、`trajectory_step` 用于追踪来源；manifest 保留原始图统计。

默认 value 标注为 `max(0, 1 - 0.01 * remaining_steps)`，例如剩余 3、2、1 步时分别
为 0.97、0.98、0.99。源断言的一步样本也标 0.99。它表示带动作成本的已知证明回报，
不是数学可证性的概率；超过 100 步而截断到 0 也不表示不可证。
默认成本与 MCTS 的默认步惩罚相同；修改成本参数后也应匹配搜索设置，不能混用含义。

## 2. 同一状态内选择动作，允许多个正确答案

每个状态保存至多 16 个候选，使用与推理共用的符号候选生成器，并强制保留证书动作。
所有候选都必须通过完整状态上的类型、替换和变量约束检查。
除证书动作外，能利用同一证书的证明计划完成**全部剩余目标**的候选也成为正例。
没有找到后续证书的合法候选只算尚未确认，不标为数学错误；有界枚举仍可能漏掉正例。
普通语料在入库前合并相同完整状态的正例见证，Scale 在同一生成图内合并；不会仅按
当前目标相同就跨假设或跨待证队列转移正例。跨图增量合并尚未实现。

候选头损失是 `-log(正例集合的总概率)`。每一行只归一化该状态自己的候选集合，
不再让别的状态中的不相关动作充当新数据的候选。普通训练和 Scale 默认候选损失权重
均为 0.25。自回归动作解码器仍使用标准下一词交叉熵，value 使用成本回报的 MSE。
多正例目前作用于候选头；下一词监督仍使用所选完整轨迹的动作。

旧语料没有候选数组时保留批内候选兼容逻辑。关闭新候选数据生成可用
`--candidate-limit 0`，但这也会恢复该兼容路径。`audit-corpus` 和
`audit-scale-corpus` 会检查候选合法性、正例索引和参考动作是否包含在正例内；
它们不重新搜索每个正例的后续证明，正例见证在数据生成时检查。

## 3. 官方验证驱动的搜索回放和在线 RLVR

`collect-replay` 只从训练集合抽取可独立发起证明的单目标状态，支持普通和分片语料。
成功轨迹编译为完整证书，经项目内核和官方 C Metamath 验证后才允许进入成功回放。
验证器缺失、拒绝、超时或异常不会成为成功奖励；搜索预算耗尽沿用未知结果屏蔽逻辑。
`reinforce` 继续训练回放中的搜索策略分布和已知 value 目标。

新增 `train-rlvr`：对同一训练目标按当前候选策略随机尝试多条路线，完全闭合并通过
官方验证得到奖励 1，预算内未完成得到奖励 0。这里 0 仅表示这次尝试未获奖励；不会
把失败目标写入 value=0 的可证性数据。候选缺失和编码限制也记录明确的未完成原因。

同组奖励减去组均值并标准化，用裁剪策略比率和 KL 约束更新候选策略。旧策略参考
是该组采样时的分布；KL 使用同一候选集合的完整分布。每组更新后再采样下一组，
不是对固定正例做额外一遍模仿训练。整个组奖励相同则不更新；任一已闭合证明出现
认证问题则跳过整组更新。时间预算是动作之间检查的软上限，认证另有超时限制。

这一版 RLVR 优化候选头和共享编码器，不直接优化自回归解码器或 value 头；共享
表示改变仍可能影响 value 输出。后续需要混入监督回放、独立校准及更多种子实验，
不能仅凭一次奖励更新宣称证明能力提升。

输出包含 `rollouts.jsonl`（候选、选择、旧概率、奖励、官方验证结果）、成功证书片段、
`rlvr.json`、tokenizer 和 `final.pt`。证书片段需与 summary 中记录的原形式系统一起加载。
新命令必须指定官方程序路径，输出使用新目录；不会覆盖输入检查点。

## 4. 按已知证明成本和结构分层抽样

新语料默认 `--sampling-mode balanced`。分组依据为：

- 已知剩余动作数：1、2–3、4–8、9+；
- 使用的规则及目标主符号；算子变量按类型归类，避免变量名制造虚假类别；
- 当前假设数量（4+ 合并）、待证目标数量（3+ 合并）、规则逻辑前提数（3+ 合并）。

训练抽样权重为 `0.25/N + 0.75/(K * n_group)`，保留 25% 自然抽样份额，另 75%
在现有分组间均衡。普通语料在训练集合上计算；Scale 在有限缓冲窗口内近似计算，
每个窗口输出相同数量的样本，保留确定性续训。验证/测试迭代不做此加权重采样。
Scale 生成入库和搜索目标选择使用分组轮换，使有限配额覆盖更多类别。

manifest 的 `supervision_histograms` 记录目标数、假设数、剩余动作数、规则前提数、
候选数、正例数和 value 分布。均衡不能创造原来没有的深证明，也不能代替前向父节点
选择的优化；条件化合一、回溯、增量索引和跨图引理池仍属于后续工作。

分组比例是可复现的初始配置，并非已证明最优。比较时应同时报告独立完整证明成功率、
官方认证结果、超时、采样分布和成本；不要把较低训练损失或参考深度当作能力证据。

## 使用和兼容性

PowerShell，在仓库根目录运行。以下命令仅为下一轮实验示例，本次没有启动长时训练。

```powershell
$env:PYTHONPATH='src;v2'
python -m neural_prover build-corpus formal/peano-pa-plus.mm outputs/trajectory-corpus --seeds 7,11 --steps-per-seed 200 --max-proof-depth 12 --max-state-tokens 1024 --max-action-tokens 1024
python -m neural_prover audit-corpus formal/peano-pa-plus.mm outputs/trajectory-corpus
python -m neural_prover train outputs/trajectory-corpus outputs/trajectory-model --epochs 1 --device cuda

python -m neural_prover collect-replay outputs/trajectory-model/final.pt outputs/trajectory-corpus formal/peano-pa-plus.mm outputs/trajectory-replay.jsonl --examples 32 --simulations 100 --external-verifier outputs/tools/metamath/metamath.exe
python -m neural_prover reinforce outputs/trajectory-model/final.pt outputs/trajectory-replay.jsonl outputs/trajectory-reinforced.pt --epochs 1
python -m neural_prover train-rlvr outputs/trajectory-model/final.pt outputs/trajectory-corpus formal/peano-pa-plus.mm outputs/trajectory-rlvr --groups 8 --group-size 4 --max-steps 24 --external-verifier outputs/tools/metamath/metamath.exe
```

`build-scale-corpus` 同样支持 `--graph-training-mode trajectories|steps`、
`--trajectory-step-penalty`、`--candidate-limit`、`--sampling-mode balanced|natural` 和
`--no-shortcuts`。`steps` 保留原来拆开的单步监督；`--no-shortcuts` 保留原证书重放路线。
新图语料需新目录，旧 manifest 不可直接续生成。旧训练数据和已训练模型不会自动升级。
旧模板扩增只读取单目标样本，跳过多目标样本，避免悄悄丢失剩余义务。

集合仍按当前目标的 alpha 标准化结论族划分，并非按完整证明骨架隔离。同一轨迹可以
贡献不同集合的记录，队列中的其他目标也可能跨集合出现。因此单步验证结果只能作诊断。
完整证明基准的训练重叠检查现扫描每个训练状态的**全部目标**，也支持 Scale 分片。
严格泛化实验仍需独立目标、证明骨架隔离及固定搜索预算。

## 本地验收

- 最终完整回归：331 项测试通过，覆盖 v1、HTPS、Scale 和 v2；另有 PyTorch 原型
  nested-tensor 提示，不影响测试结果。
- PA+ 种子 7、100 次图生成尝试：643 条训练/验证/测试记录；122 条多目标状态，
  60 条具有多个正例，value 为 0.83–0.99，已知剩余动作最多 17。
- 所有参考动作和候选通过内核审计；小模型完成一轮 CPU 监督训练。
  40 条 Scale 样本也通过审计，并完成两步包含候选损失的 CPU Scale 训练。
- PA+ RLVR 的 8 次尝试中有 4 次通过官方验证；两组分别全成功和全失败，所以按设计
  跳过更新。这只是流程检查，不是能力评测。
- 极小形式系统中同一可证目标有一步和两步路线，给定一步预算，16 次尝试得到
  8 个官方认证成功及 8 个预算未完成，完成一次真实策略更新，检查点中 33 个参数张量发生
  变化；另有一次认证 MCTS 回放入库。
- 工程结果在被 Git 忽略的 `outputs/proof-training-smoke-20261011/` 和
  `outputs/proof-training-official-20261011/`，不与历史长时训练结果混合。
