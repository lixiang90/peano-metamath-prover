# PA+ 神经训练与闭环推理

本文说明扩展 PA+ 形式库和定理生成器如何进入神经策略的词表、训练样本与证明
搜索。所有神经动作仍须由符号内核枚举和验证；模型不会获得新增公理。

## 1. 结构化词表

状态在 `<STATE>` 后加入可选的 PA+ 上下文：

```text
<PA_PLUS_CONTEXT>
  <DEFINITIONS> ... <END_DEFINITIONS>
  <TARGET_HINTS> ... <END_TARGET_HINTS>
  [<GROUND_TERMS>]
<END_PA_PLUS_CONTEXT>
```

定义部分只列出当前状态实际涉及的定义谓词。目标提示不是自然语言，而是从35个
非逻辑 `statement` 中按固定符号重合度选择的形式公式。不存在自由项变量时加入
`<GROUND_TERMS>`。生成式定义桥共享 `<DEFINITION_BRIDGE>` 和 `<UNFOLD>`/`<FOLD>`
标记；替换中的规则变量编码为 `<V:类型:序号>`，避免把局部变量名误当成语义。

词表采用追加升级，旧 token ID 和对应模型权重保持不变。旧版 tokenizer 文件仍
可加载。升级命令会丢弃旧 optimizer 状态，因此升级后的 checkpoint 用于重新训练
或微调，而不是无缝恢复同一个优化器轨迹：

```bash
python -m neural_prover upgrade-pa-plus \
  OLD.pt OLD-tokenizer.json formal/peano-pa-plus.mm \
  formal/pa-plus-definitions.json NEW.pt NEW-tokenizer.json \
  --bounded-nat-max 2 --max-target-hints 3
```

## 2. 语料和训练

`build-corpus` 现在把定义桥、闭式实例、目标反向引导和定义包装限流参数传给生成器，
并为每条样本记录 `generation_kind`、`guidance_target` 和 `definition_support`。一个
中等本地实验可从以下配置开始：

```bash
python -m neural_prover build-corpus \
  formal/peano-pa-plus.mm outputs/pa-plus-corpus \
  --seeds 7,11 --steps-per-seed 300 \
  --definition-catalog formal/pa-plus-definitions.json \
  --bootstrap-definitions --definition-coverage-weight 3 \
  --bounded-nat-max 2 --ground-instances-per-predicate 1 \
  --target-guidance-weight 4 \
  --max-definition-only-search-per-predicate 1 \
  --max-state-tokens 1024 --max-action-tokens 512

python -m neural_prover train outputs/pa-plus-corpus outputs/pa-plus-model \
  --device cuda --epochs 3 --batch-size 16 \
  --candidate-loss-weight 0.35
```

训练损失包含自回归动作损失、价值损失和候选动作对比损失。候选损失把批内相同
动作视作多正例，避免把重复合法动作当成负例。PA+ tokenizer 启用时，采样器提高
稀缺闭式实例的权重；普通 tokenizer 不启用该采样。可用
`--no-pa-plus-balanced-sampling` 做消融。

## 3. 推理和证书

加载 PA+ tokenizer 时，环境会自动：

1. 从68个保守定义生成136条展开/折叠桥并用内核构造其证明；
2. 为项变量加入 `0` 到配置上界的规范自然数闭项；
3. 按当前目标与35个形式目标的符号接近度调整断言枚举顺序；
4. 将模型选择的生成式桥展开为原始语法规则、`df-*`、`bi` 和 `mp` 标签。

所以 `gen_df_*` 只是搜索宏，不进入最终可信基。最终证书既由项目内验证器重放，
也可交给官方 Metamath 实现交叉验证。

## 4. 2026-08-31 本地 GPU 工程验证

RTX 3060 Laptop GPU 上完成了一次小型但非纯 CPU 的闭环验证：

- 语料共616例：127条源证明、136条定义桥、136条有界闭式实例、217条引导搜索；
- 217条引导搜索合计触达全部35个形式目标，词表大小为882；
- 64维、2层 encoder/decoder 模型训练3轮，验证总损失从5.9908降至3.8880；
- 验证 token accuracy 从25.77%升至36.94%，候选准确率从6.90%升至15.52%；
- 混合策略在一个未直接写入源库的定义桥实例上1步闭合；导出的33标签证书不含
  生成式桥标签，并同时通过项目内核与官方 Metamath；
- 推理候选中确认出现 `0`、`S 0`、`S S 0` 三个有界自然数项。

这个实验只证明接线、CUDA 训练和证书闭环可用。小模型的纯神经策略把该直接桥排
在17个候选中的第14位，因此不能声称 PA+ 解题能力已经提升。下一步应在固定未见
目标、固定搜索预算和多随机种子下，对照关闭目标提示、闭式实例、均衡采样及候选
损失，报告 solve rate、搜索节点数和外部验证通过率。
