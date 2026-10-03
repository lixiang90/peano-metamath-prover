# PA+ v2 开发目标与验收清单

本目录是独立的新版本，复用原项目 PA+ 语法、生成规则与 Metamath 验证内核。已有 v1 检查点与正在运行的训练保持独立。本文是实现和验收依据；工程 smoke 不代表困难数学问题的证明能力。

## 1. 用户要求

1. 主模型采用 decoder-only 自回归 Transformer，RoPE 位置编码。
2. 引理编码器同样采用 decoder-only 自回归 Transformer。引理末尾的一个或多个 summary slots 产生连续向量，在引理出现的位置与普通 token embedding 按顺序拼接，进入同一个主干。两个模型均无 cross-attention。
3. 训练数据来自 PA+ 公理及保守定义的随机规则生成；不添加人类数学语料或将开放猜想视为事实。
4. 预算内选择重要定理构建可检索、可调用的库。只接收通过证明重放的对象，并记录理论版本、证明、假设、变量类型与 distinct-variable 约束。
5. 实现结构化 Agent 草稿纸，明确区分 conjecture、verified fact、持久库与局部笔记。
6. 推理轨迹以显式 action/observation 文本留在上下文，包含失败和恢复。引理向量只编码已知引理，不承担隐向量思维链。
7. **预训练采用标准下一 token 预测**：对生成语料的全部可预测文本 token 使用移位交叉熵。Agent SFT 可另用 action-only mask。
8. **支持 RLVR**：采样 Agent 证明轨迹，依据独立证明验证的终局信号给予奖励，通过 policy gradient 更新模型；不能把成功样本 SFT 冒充 RLVR。预算耗尽得未成功奖励，不能解释成定理为假。
9. 完成实现、回归、端到端验证后提交并经用户指定代理推送 GitHub。

## 2. 架构与边界

### 模型与文本

采用常见 Llama 风格的 pre-norm RMSNorm、RoPE causal self-attention、SwiGLU。主模型与引理模型独立参数。固定 UTF-8 byte tokenizer 使用 PAD/BOS/EOS/MEM 加 256 字节，避免库扩充改变词表；这是一项工程选择，长公式的 token 效率需后续评测。

每次读入引理时，默认只向主模型上下文写入引理 ID/出处等元数据，再放置 `lemma_slots` 个连续 MEM 位置。引理编码器读取完整声明（包括类型、假设和 `$d`），最后追加可学习 summary slots；slots 只因果读取前面的内容。SEARCH/READ 的完整声明保留在符号环境和审计日志中，不重复作为主模型的普通 token，避免文本旁路。模型以等长向量块替换这些 MEM embeddings，因而 token、label、position 保持对齐。内存位置不是语言预测目标，但其输出可以预测后续文本。外部库中的精确符号声明与证明始终保留，向量不作为逻辑真值。Python 提供 text/both 模式用于后续消融。

参考实现依据：[Meta Llama](https://github.com/meta-llama/llama-models)、[Transformers Llama 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)。用户要求优先于早先讨论过的双向引理编码器方案。

### 符号内核与 Agent

采用固定证明目标上下文的前向证明工作区：所有事实都携带在此上下文中可重放的证明。`APPLY` 应用源规则，`USE` 应用已证明的引理，检查精确类型、所有前提与 `$d` 后才提交新事实。条件引理只能在前提已有证明时使用。最终导出的证明展开引理宏，不增加公理。

`PROPOSE` 记录待证子目标；`SAVE` 保存已验证事实到草稿纸；`SEARCH` 返回候选库 ID；`READ` 读取声明并插入引理编码；`BACKTRACK` 恢复先前工作区；`FINISH` 对目标证书独立重放。失败操作保留在轨迹中，但不增加可信事实。所有资源超限明确返回 unknown/budget-exhausted。

### 定理库

采用可审计的近似效用评分：证明长度/潜在节省、训练集复用频率、符号覆盖多样性，受条目数及序列化字节双预算约束。选择不是最优压缩的理论保证；验收检查的是硬预算与来源正确性。检索支持确定性词法回退和引理向量相似度，向量索引记录编码器指纹；权重变化必须刷新，不能静默混用旧索引。训练期间可以根据使用记录重新排序、重新选择并刷新索引。

引理依赖不触发递归神经编码；首版采用声明级独立编码，详细研究依据和后续实验路径见 [引理依赖与循环计算](lemma-dependencies.md)。

## 3. 数据算法

1. 用固定随机种子在 PA+ 上运行现有经过验证的规则组合生成器，增加定义桥接、深父节点和复用倾向；混入有限数量的短源规则随机实例作为一步起步课程，仍由同一形式系统证明。
2. 导出完整证明 DAG，展开为可独立重放的源标签证书；拒绝无证书或超限样本。
3. 按规范化目标和证明骨架分组划分 train/validation/test；定理库只从 train 构建，评估前冻结。排除目标本身；不能用测试证明指导候选生成。
4. 将证书变成 Agent teacher actions；构造 scratchpad SAVE/READ/USE、有界失败、BACKTRACK 与恢复事件。错误尝试仅改变文本历史，不制造正例。
5. 保存完整文本作为标准 NTP 语料；同时保存结构化 episodes，重放生成 action-only SFT 样本。所有标签由可复现的符号算法产生。
6. 清单记录 seed、理论哈希、生成配置、划分规则、过滤原因、证明长度与动作覆盖。不把过滤掉的长证明默默算作成功。

## 4. 训练与推理

- NTP：全生成文本下一 token CE，padding/memory 控制位置不计 loss。
- Agent SFT：在明确 action/observation 轨迹上训练下一动作，环境返回本身可作为 NTP 语料。
- 引理 encoder：被读取的引理向量与主模型联合反传；检索最初使用词法/冻结向量，可定期用当前编码器刷新。离散 top-k 不声称可端到端反传。
- RLVR：按同一目标采样一组轨迹；终局通过验证奖励为 1，其他为 0。组内优势归一化，PPO/GRPO 风格 clipped policy objective，可加 reference-policy KL；相同奖励组无相对学习信号，应跳过并报告。候选动作策略由自回归序列分数构成 categorical 分布；默认长度归一化并混合均匀探索（详见运行指南），rollout 与更新必须使用同一候选集、温度与探索参数。记录 old log-prob，不能重新采样候选集用于旧轨迹更新。损失按动作平均，非逐轨迹等权。
- 推理：可从当前符号状态生成有限合法候选，通过主模型自回归概率评分/采样；这只限制动作空间，不提供目标的 teacher proof。保留失败轨迹，明确步数、上下文、库读取和证明大小预算。神经评分失败/输入超长应返回可诊断状态。

## 5. 验收矩阵

| 要求 | 必须提供的证据 | 状态 |
|---|---|---|
| 主/引理 decoder-only RoPE | 因果遮罩、位置、padding、1/多槽测试 | 通过：test_model.py |
| embedding 顺序融合 | 插入位置、连续块完整性、引理 encoder 梯度测试 | 通过：模型/训练测试，NTP/SFT 实际反传 |
| 定理库预算/可信性 | 数量/字节上界、tamper/theory mismatch、条件前提/DV 测试 | 通过：test_library.py、test_memory.py |
| 草稿纸与显式轨迹 | SAVE/READ/USE、失败不污染、BACKTRACK 恢复测试 | 通过：test_agent.py；30 次草稿、1 次全局库 USE |
| 随机 PA+ 数据 | 可复现生成、split 防泄漏、证书重放、动作覆盖 | 通过：48 条，独立进程重建哈希一致 |
| NTP 与 SFT | 移位 CE 对照、训练 loss 有限、两个模块梯度 | 通过：12 步 NTP、16 步 SFT，过滤数明确 |
| RLVR | 真实奖励、采样概率一致、参数更新、零优势组处理 | 通过：16 次采样、9 次认证、2 组实际更新 |
| 完整闭环 | CPU generate → NTP/SFT → library → Agent/RLVR → 官方 Metamath | 通过：48/48 教师认证；小型测试集 3/5 成功 |
| v1 兼容与发布 | 全套回归、generator smoke、打包、git diff、GitHub push | 本地 252 项、旧生成器、wheel/sdist 独立安装通过；远端交付见提交 |

实际命令、计数和限制见 [验证记录](validation.md)。3090 当前训练未被本轮开发占用；CPU smoke 仅验证实现闭环。后续规模化实验应比较无库、文本库、向量库、无失败轨迹、SFT、RLVR 等消融，在相同搜索预算与独立测试族上报告认证成功率、证明长度和总计算成本。
