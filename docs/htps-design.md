# 增强 PA 下的 HTPS 数据、训练与推理

## 1. 目标与可信边界

本版本借鉴 HyperTree Proof Search，但不复制 Evariste 的专用 Equations 环境。
所有状态、动作、替换、类型和 `$d` 条件仍由项目的 Metamath 内核解释。搜索成功
只是候选结果；只有动作序列能编译成证明，并被独立验证器完整重放，才计为解决。

连续隐式思维只改变 policy、critic、lemma gate 和 halt 的数值；它不能创建
事实、修改形式状态或绕过证书验证。

## 2. 认证前向证明 DAG

`htps_prover.forward` 在已有质量控制生成器上增加三项约束：

1. 保存每个派生节点、规则、替换、前提映射和递归证明骨架；
2. 按递归证明骨架 SHA-256 分配 train/validation/test，使同一推理模板的类型
   保持变体不会跨 split；
3. 对每个 seed 的最深若干依赖族执行“导出 Metamath → 重新解析展开后的源库 →
   逐条验证 `$p`”的独立回放。

输出包括：

- `proof_dag_nodes.jsonl`：完整生成 DAG 元数据；
- `policy_{split}.jsonl`：目标到下一条认证规则的监督数据；
- `lemma_{split}.jsonl`：由实际父证明实例化而来的中间引理动作；
- `tokenizer.json` 与 `manifest.json`。

若从旧 checkpoint 继续训练，生成时必须传入其 tokenizer。构建器只在词表末尾
追加 lemma 控制符，保持所有旧 token ID 不变；随机重新构建相同“符号集合”的
词表仍可能破坏 checkpoint 对齐，因而不能替代这一步。

内部 DAG 节点不是凭空生成的公式：它来自一个已经构造并可回放的证明依赖。
因此它适合为 lemma proposal 提供正样本。后续仍应按“是否缩短最终证书、是否被
多个分支复用”进一步筛选高价值引理。

## 3. 超图语义

一个 goal node 是包含单一当前目标的 `ProofState`。普通规则应用形成超边：

```text
g --t--> {g1, g2, ..., gn}
```

在 OR 层只需选择一个 tactic；在 AND 层必须解决该 tactic 的所有子目标。相同的
单目标状态按完整假设、目标、变量类型和 `$d` 约束共享，形成 transposition。

节点状态递归定义为：

- solved：至少一条超边的全部孩子 solved；
- invalid：节点已经展开，且每条超边都含 invalid 孩子或没有合法边；
- open：尚可继续展开。

备份值使用孩子值的乘积，并施加可配置 depth decay。PUCT 结合模型 prior、搜索
Q、访问次数和 virtual loss。每轮先连续选择多棵部分证明 hypertree，virtual
loss 促使它们覆盖不同分支，再统一扩展去重后的 frontier goals；因此不是只走
一个完整多目标状态。

## 4. 受保护的引理超边

引理动作具有顺序依赖：

```text
Γ ⊢ G
  -- propose L -->
      1. Γ ⊢ L
      2. Γ,L ⊢ G
```

`guarded=True` 的超边一次只开放第一个未解决孩子。只有第一项 solved 后，搜索才
能展开第二项。提取证明时使用前序顺序重放，原 `LemmaBackwardEnvironment` 的
commit marker 再次保证未证明的引理不会成为活动假设。

## 5. 训练目标

监督阶段联合训练：

- forward DAG 的正式规则动作；
- 从实际父依赖得到的 lemma proposal；
- 状态价值；
- 连续 latent thought 与 halt head 的 ponder 正则。

在线阶段从 HTPS 超图提取：

- policy：只标注根的最小已认证证明 hypertree，使用 one-hot 最短证明边；
- critic：solved=1、invalid=0；访问充分的内部节点使用最终 HTPS 软 Q；
- censored：预算耗尽且访问不足的节点不提供伪负标签。

候选策略损失在同一状态的有限 kernel-valid 动作集合上计算 soft cross entropy；
价值使用 MSE，latent halt 继续接收任务梯度和 ponder cost。

`closed-loop` 当前实现为单机同步 actor/trainer 版本：每轮搜索、认证、合并有限
容量 replay、训练新 checkpoint，再进入下一轮。数据格式与组件边界允许以后将
actor 扩展为异步多进程或多 GPU，而不改变证书边界。

## 6. 评估

必须同时报告：

- certificate solve rate / pass@k；
- 抽象动作数与最终 Metamath certificate steps；
- goal nodes、hyperedges、expansions、transposition hits；
- guarded lemma 数量及实际进入最终证明的数量；
- wall time 和神经候选评分数；
- 按证明骨架隔离的 ID/OOD 结果。

`evaluate` 命令只把成功编译并重放的证书计为 solved。模型声称成功、搜索图标记
solved、或抽象动作到达空状态，任何一项单独都不足以进入最终统计。

使用 `--randomize-search` 时，每题会从 CLI 给出的上界内确定性采样 simulation、
expansion、branching、PUCT、temperature 和 depth decay。实际取值逐题写入结果，
同一全局 seed 可完整复现；不启用时只为每题派生独立的并列分支随机种子。

## 7. 当前限制

- 神经调用已经按一个状态的候选集合批量化，但多个 frontier goal 的编码尚未合并
  为一个跨节点 GPU batch；这是下一项主要性能工作。
- lemma 正样本当前来自直接父依赖，尚未执行全 DAG 的最优 cut/复用收益搜索。
- 同步闭环用于本地验证算法；大规模训练需要异步 actor 队列、模型版本戳和陈旧
  replay 权重。
- 当前只实现 pass@1；pass@k 需要同一目标的多次独立搜索调度。
