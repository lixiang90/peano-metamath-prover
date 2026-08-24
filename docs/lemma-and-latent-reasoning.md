# 中间引理与连续潜在思维

本页描述 2026-08-24 新增的工程原型。它已经通过内核、搜索、回放和证书回归
测试，但尚未经过规模训练，也不能据此声称解题率有所提高。

## 1. 可验证的中间引理动作

词表在现有 token ID 之后追加三个控制 token：

```text
<PROPOSE_LEMMA> <LEMMA> ... <END_LEMMA>
```

旧 tokenizer 从磁盘加载时保持原样；`upgraded_for_lemma_actions()` 只追加缺少的
token，因此已有 token ID、旧 embedding 行和旧策略头行都不会移动。

对当前目标 `Γ ⊢ G` 提议 `L` 时，符号环境执行严格的 cut 生命周期：

```text
Γ ⊢ G
  ├─ 先证明 Γ ⊢ L
  └─ L 通过后才允许证明 Γ,L ⊢ G
```

内部 commit 标记确保 `L` 在其证明的全部子目标关闭前不能作为假设使用。候选引理
必须是良类型的 `|-` 公式，并受数量、大小和待处理层数限制。候选来自有限符号构造，
神经网络只负责排序，不能把任意连续向量直接写进证明状态。

证书编译器把 cut 内联：先编译 `L` 的证明，再把该证明作为局部宏展开到后续分支。
最终 Metamath 证书只包含原形式库中的合法标签，不包含 `<PROPOSE_LEMMA>` 或内部
commit 标记。

## 2. 隐式连续向量思维链

旧模型文件 `model.py` 保持不变；新模型定义在 `latent_model.py`。编码证明状态后，
模型在动作选择前递归更新隐藏向量：

```text
h0 = pool(Encode(state))
ht = LayerNorm(gate * tanh(Wc h0 + Wr h{t-1}) + (1-gate) * h{t-1})
halt(ht) -> continue / act
```

隐向量不对应自然语言 token，也不被保存为证明步骤。它只影响价值估计、有限候选
排序以及“普通动作/引理动作”的门控先验。推理阶段可由停止头提前终止，且始终受
最大思考步数约束；训练阶段用策略/价值梯度和很小的 ponder cost 学习表示与停止
行为，不需要文本思维链标签。

## 3. 兼容升级和闭环运行

从现有检查点派生新检查点，不会覆盖旧文件：

```bash
python -m neural_prover init-latent \
  outputs/model/final.pt outputs/corpus/tokenizer.json \
  outputs/model-latent/initial.pt outputs/model-latent/tokenizer.json \
  --max-thought-steps 6 --min-thought-steps 1
```

命令会复制所有形状兼容的旧权重，并逐行复制旧词表 embedding 和策略输出；仅新增
token 行、潜在递归层、停止头和引理门控头随机初始化。原检查点和 tokenizer 不会
被改写。

对一个隐藏参考证明的 benchmark 题运行新闭环：

```bash
python -m neural_prover prove-decomposed \
  outputs/model-latent/initial.pt outputs/model-latent/tokenizer.json \
  outputs/benchmark-v2.json CASE_ID formal/peano-number-theory.mm \
  --simulations 100 --max-depth 20 --branching 24 \
  --certificate outputs/decomposed.mm
```

闭环顺序为：符号内核生成普通和引理候选，连续潜在层规划，模型排序，MCTS 执行
离散动作，最后编译并验证证书。MCTS replay 已支持引理动作序列化，因此后续可用
认证轨迹训练候选策略；`latent_policy_value_loss` 提供包含停止头 ponder loss 的
监督/RL 热启动损失。

## 4. 当前边界与后续验收

- 新增权重尚未训练，直接从旧检查点升级只验证兼容性，不代表能力提升；
- 当前引理候选依赖有限构造与浅层可证提示，规模评估后需优化召回率和去重；
- 必须分别消融无引理、仅离散引理、固定潜在步数和可学习停止头；
- 在相同或明确计价的搜索与计算预算下，以 certified solve rate、深度分桶、
  平均思考步数和外部验证通过率作为验收指标。
