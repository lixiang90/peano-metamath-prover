# v2.1 结构化反馈与可执行草稿纸

此功能只扩展独立的 `v2_1/pa_prover_v21`。v1、v2 和 PA+ 公理文件保持原样。
本轮只进行回归、证书验证和微型梯度更新，不进行完整训练，不宣称数学能力提升。

## A/B/C 模式

| 模式 | CLI 值 | 输入与执行方式 |
| --- | --- | --- |
| A | `baseline`，默认 | 原有状态、动作历史和有限候选评分；原搜索路径保持独立 |
| B | `feedback` | 连续 `ENGINE/MODEL` 轨迹；每次 APPLY 后读取结构化执行反馈 |
| C | `scratchpad` | B 加上一层短 TRY、COMMIT、BACKTRACK 和尝试记忆 |

B/C 使用相同的原有高层动作生成器：INDUCT、HAVE、STRENGTHEN、USE、UNFOLD。
本轮没有实现任意辅助公式生成、自然语言解释或循环编码器。神经模型仍给有界合法
命令评分；C 增加的是对试探和恢复的选择，不是无约束地生成一篇“思考文章”。

每个 ENGINE 记录包括完整未解决目标（含被阻塞的续接目标）、局部事实及其假设、
变量类型、不同变量约束、剩余预算、关闭的子目标以及逐目标的经典求解结果。
`UNKNOWN` / `UNKNOWN_WITHIN_BUDGET` 均不表示命题为假。`ROOT_READY` 只表示本地
构造完成；最终导出且通过官方 Metamath 验证，并在预算内结束，才记为 `certified`。

MODEL 只能选择执行命令，不能填写或覆盖 ENGINE 反馈。角色按程序生成，不信任模型
自行声称的证明状态。审计用证明 DAG、执行回执以及未来事件不进入模型上下文。

## 短试探、回退和记忆

TRY 在隔离的 session 中执行一个高层动作，然后运行短时间经典闭包。首版默认
闭包预算 0.25 秒、每条轨迹最多 4 次试探、最多 24 个交互命令。普通证明深度限制
仍是 5 个高层动作；COMMIT/BACKTRACK 不计入逻辑深度，但计入交互和墙钟预算。

试探后必须选择 COMMIT 或 BACKTRACK，不能叠加未决试探。COMMIT 接受当前分支；
如果短试探尚未完成，提交后会用正常闭包预算继续求解，仍受总墙钟预算限制。
BACKTRACK 恢复原目标、证明框架、事实、变量分配和作用域。试探中的归纳假设、
尚未证明的引理及局部证明不会自动进入原分支。闭包缓存也按分支隔离。
HAVE 的续接目标始终在引理完成之前阻塞。

回退保留“在这个状态尝试过什么、观察到什么”的记录，不把记录当成逻辑事实。
最近 8 条记录显式展示在反馈中，完整过程保存在连续轨迹中。同一状态的同一个
TRY 不重复调用；对应的 APPLY 仍然可选，因为短预算失败不应永久淘汰一个动作。
所有模型评分、试探、恢复、证书导出和最终验证使用同一个总墙钟预算。单次 Python
操作可能有少量超时尾部，但超过总预算的证明不计成功；评测另有进程 watchdog。

## 数据与训练

问题仍从原有自生成证明图与有限算术猜想文法产生，在教师搜索之前按结构族划分。
所有反馈来自真实执行。启发式和可选的学习教师，以及随机试跑，可以产生成功的
连续轨迹；有多个合法成功路径时均保留，不伪造唯一解释。

生成器同时保存三类样本：

* `policy`：具有相同完整可见前缀和候选集的多正例。
* `trajectory`：从初始状态到成功结束的连续交互，包含实际发生的试探和回退。
* `value`：由独立试跑估计的预算内完成频率，包括失败状态。

A 保留原训练数据及目标。B/C 的 NTP 只抽取连续轨迹，用普通的全序列下一词
交叉熵训练；SFT 混合多正例策略、连续轨迹和价值样本，其中连续轨迹只监督 MODEL
内容，ENGINE 内容作为条件输入。UTF-8 字节跨度与掩码对齐，命令以换行结束，
候选评分与训练使用同一个序列格式。

RLVR 从语料 manifest 读取模式和交互限制，仍从头执行独立的 on-policy 试跑。
TRY、COMMIT、BACKTRACK 都属于真实采样的分类决策，保存其候选集和完整概率。
不把 best-first 教师轨迹当 PPO 样本；全成功或全失败的零优势组跳过策略更新。
只由最终官方证书给出完整证明奖励，草稿长度、猜想数量本身没有正奖励。

完整前缀超出上下文时明确报告 `context_overflow` / `unknown_context`，不截掉未
解决目标，也不把上下文失败当作可用的价值负例。首版保存完整历史，长轨迹成本
可能较高；后续可单独研究具有状态保持保证的压缩。本轮尚未实现长轨迹分块。

## 审计与泄漏隔离

每次经典闭包保存不可见的执行回执：关闭子目标的证明 DAG、结构分解以及新鲜变量
上下文。`audit-strategy` 重放这些记录，检查每一步的可见状态、作用域、候选集和
模型前缀；同时复验成功轨迹的最终官方证书。重放只允许同样的新鲜变量扩展，
不允许修改已有变量类型或任意增加 `$d` 条件。

耗时和未成功搜索是已记录的运行观察，不能像证明一样确定性重放；审计不会把
“超时”解释成否证。训练样本还必须与 manifest 绑定的成功执行记录一致。
既有跨轮划分登记、冻结引理快照、验证／测试证明不得晋升训练库等规则继续适用。

## 使用

以下为命令模板，运行时使用新的输出目录。训练必须显式调用，不会由证明命令启动。

```text
python -m pa_prover_v21 generate-strategy formal/peano-pa-plus.mm outputs/v21-b --metamath outputs/tools/metamath/metamath.exe --reasoning-mode feedback --count 24 --seconds 10
python -m pa_prover_v21 generate-strategy formal/peano-pa-plus.mm outputs/v21-c --metamath outputs/tools/metamath/metamath.exe --reasoning-mode scratchpad --count 24 --seconds 10 --trial-seconds 0.25 --max-trials 4 --max-interactions 24
python -m pa_prover_v21 audit-strategy formal/peano-pa-plus.mm outputs/v21-c --metamath outputs/tools/metamath/metamath.exe
python -m pa_prover_v21 train-policy formal/peano-pa-plus.mm outputs/v21-c outputs/v21-c-ntp --stage ntp --steps 100 --device cuda
python -m pa_prover_v21 train-policy formal/peano-pa-plus.mm outputs/v21-c outputs/v21-c-sft --stage sft --steps 100 --device cuda --checkpoint outputs/v21-c-ntp/policy.pt
```

`train-policy` 和 `rlvr` 从语料读取模式；`generate-strategy`、`expert-iterate`、
`prove-strategy`、`evaluate-strategy` 接受显式 `--reasoning-mode`，默认均为 A。
同一次比较可以使用 `heuristic-a,heuristic-b,heuristic-c` 或
`learned-a,learned-b,learned-c`。纯 ATP、传统归纳组始终沿用经典路径。

```text
python -m pa_prover_v21 evaluate-strategy formal/peano-pa-plus.mm outputs/v21-c outputs/v21-abc --metamath outputs/tools/metamath/metamath.exe --vampire outputs/tools/vampire-5.0.0/vampire.exe --methods atp,induction,learned-a,learned-b,learned-c --checkpoint outputs/v21-a-sft/policy.pt --checkpoint-feedback outputs/v21-b-sft/policy.pt --checkpoint-scratchpad outputs/v21-c-sft/policy.pt --split test --seconds 10 --device cuda
```

未提供某模式专用 checkpoint 时回退到 `--checkpoint`，可用于同权重输入消融；
正式训练效果比较应提供各自训练的模型，并匹配训练资源。报告保存各模型散列、
模式、交互限制、墙钟耗时、经典调用与证明成功率。B/C 另记录试探、回退、模型评分
调用数和字节开销（非 FLOPs）。所有组共享冻结的问题集、引理库及总证明时间。
微型验收结果只能说明管线正确运行，不能作为 B/C 数学能力提高的证据。

## 本轮验收（2026-10-12）

最终全仓库回归为 **377 passed**，包含官方证书、分支变量与假设隔离、回退恢复、
未证明 HAVE 阻塞、连续轨迹因果前缀、反馈篡改拒绝、MODEL 掩码、候选序列概率
一致性、上下文溢出以及 B/C 生成—审计—微型训练接口测试。微型网络宽度 16、
1 层；每个模式的测试执行 NTP/SFT 各一步更新和一组两次 RLVR 试跑，未启动完整训练。
可编辑安装、wheel 中五个包与新文档的检查、原生成器 100 步 smoke 均通过。

独立六题自生成语料位于 `outputs/v21-bc-acceptance`：5 道训练题、1 道验证题，
没有测试题；共 8 条策略样本、10 条连续轨迹和 6 条价值样本。审计重放 4 个
证明档案、22 条交互记录，并复验 12 份成功轨迹证书；价值样本包含 0 和 1。
真实回退后成功的路径由专项测试额外覆盖；这批小语料不提供能力或泛化证据。
同一验证题还完成纯 ATP、传统归纳、启发式 A/B/C 的 5 秒同预算进程联调，报告在
`outputs/v21-bc-comparison/report.json`。此处没有训练后的模型能力比较。

首轮全仓库测试曾有一份散列未改变的证书出现本机官方复验结果不一致；随后三次
独立复验及最终全仓库回归均通过。没有确认该瞬时异常的原因，验证门槛仍保持
失败即拒绝。首次日志 `outputs/v21-bc-full-tests.log` 和最终日志
`outputs/v21-bc-final-tests.log` 保留供追踪。其他记录包括
`outputs/v21-bc-acceptance-audit.log`、`outputs/v21-bc-wheel.log` 和
`outputs/v21-bc-generator-smoke.log`。实验产物及模型均留在被忽略的 `outputs/`。
