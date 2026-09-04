# PA+ 神经训练与闭环推理

本文说明扩展 PA+ 形式库和定理生成器如何进入神经策略的词表、训练样本与证明
搜索。所有神经动作仍须由符号内核枚举和验证；模型不会获得新增公理。

本页于 2026-09-04 按 `b911e9c` 核对；实验数字仍对应 2026-08-31，不表示重新训练。

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
非逻辑 `statement` 中按公式固定符号重合度选出的至多3个目标标签；完整公式
保存在 tokenizer 元数据中，并非原样追加进状态序列。可用 `--max-target-hints`
调整标签提示上限。状态变量类型表不含 `term` 元变量时加入
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

应给输出选择新路径，以保留原文件。升级支持基础和 latent checkpoint，但只完成
词表/权重扩容，不训练新增行，也不保证动作编码改变后行为完全不变。基础 `train`
始终从头建模，没有加载升级 checkpoint 的续训参数；升级后的 PA+ 微调入口仍需
另行接线，不能直接用下方 `train` 命令声称是在续训 `NEW.pt`。

## 2. 语料和训练

`build-corpus` 现在把定义桥、闭式实例、目标反向引导和定义包装限流参数传给生成器，
并为每条样本记录 `generation_kind`、`guidance_target` 和 `definition_support`。
以下是第4节小型工程验证的实际配置；先在仓库根目录安装 `.[neural]`，或在已装
PyTorch 的 PowerShell 中设 `$env:PYTHONPATH='src'`。多行示例使用 Bash 续行语法。

```bash
python -m neural_prover build-corpus \
  formal/peano-pa-plus.mm outputs/pa-plus-corpus \
  --seeds 7,11 --steps-per-seed 300 \
  --max-proof-depth 8 --max-ast-depth 64 --max-variables 24 \
  --definition-catalog formal/pa-plus-definitions.json \
  --bootstrap-definitions --definition-coverage-weight 3 \
  --bounded-nat-max 2 --ground-instances-per-predicate 1 \
  --target-guidance-weight 4 \
  --max-definition-only-search-per-predicate 1 \
  --max-state-tokens 1024 --max-action-tokens 512

python -m neural_prover train outputs/pa-plus-corpus outputs/pa-plus-model \
  --device cuda --epochs 3 --batch-size 16 \
  --d-model 64 --layers 2 --learning-rate 0.0005 \
  --candidate-loss-weight 0.35
```

其余模型参数取 `TrainingConfig` 默认值：4个注意力头、FFN 512、dropout 0.1、
seed 7。checkpoint 使用同批语料的 `outputs/pa-plus-corpus/tokenizer.json`；
基础训练不会在模型输出目录额外复制 tokenizer。

训练损失包含自回归动作损失、价值损失和候选动作对比损失。候选损失把批内相同
动作视作多正例，避免把完全相同的动作当成负例；不同但同样合法的动作仍可能被
当成负例。批内候选未逐个检查对当前状态是否合法，因此这里的 candidate accuracy
只是批内检索指标，不是内核合法候选上的准确率。PA+ tokenizer 启用时，采样器提高
稀缺闭式实例的权重；普通 tokenizer 不启用该采样。可用
`--no-pa-plus-balanced-sampling` 做消融。

## 3. 推理和证书

`TransformerPolicy` 构造时调用 `environment.configure_from_tokenizer(tokenizer)`，
而非仅加载 tokenizer 文件就配置所有环境。启用完整目录时会：

1. 从68个保守定义生成136条展开/折叠桥并用内核构造其证明；
2. 为项变量加入 `0` 到配置上界的规范自然数闭项；
3. 按当前目标与35个形式目标的符号接近度调整断言枚举顺序；
4. 将模型选择的生成式桥展开为原始语法规则、`df-*`、`bi` 和 `mp` 标签。

所以 `gen_df_*` 只是搜索宏，不进入最终可信基。最终证书既由项目内验证器重放，
也可交给官方 Metamath 实现交叉验证。

注意桥结论是 `P → expansion` 或 `expansion → P`。它能直接闭合相应蕴含式，
并不是任意目标 `P` 都能一步折叠成功；使用时可能仍需 `ax-mp` 和前提证明。

## 4. 2026-08-31 本地 GPU 工程验证

RTX 3060 Laptop GPU 上完成了一次小型但非纯 CPU 的闭环验证：

- 语料共616例：127条源证明、136条定义桥、136条有界闭式实例、217条引导搜索；
- 217条搜索记录的 `guidance_target` 标签合计覆盖35个目标，词表大小为882；
  这不同于按最佳公式相似度≥0.25统计的 `target_statements_touched`，两次生成
  运行该指标各为34/35；二者都不表示目标被证明；
- 64维、2层 encoder/decoder 模型训练3轮，验证总损失从5.9908降至3.8880；
- 验证 token accuracy 从25.77%升至36.94%，候选准确率从6.90%升至15.52%；
- 混合策略在一个未直接写入源库的定义桥实例上1步闭合；导出的33标签证书不含
  生成式桥标签，并同时通过项目内核与官方 Metamath；
- 推理候选中确认出现 `0`、`S 0`、`S S 0` 三个有界自然数项。

这个实验只证明接线、CUDA 训练和证书闭环可用。小模型的纯神经策略把该直接桥排
在17个候选中的第14位，因此不能声称 PA+ 解题能力已经提升。下一步应在固定未见
目标、固定搜索预算和多随机种子下，对照关闭目标提示、闭式实例、均衡采样及候选
损失，报告 solve rate、搜索节点数和外部验证通过率。

原始配置、指标和 checkpoint 位于 Git 忽略目录
`outputs/pa-plus-neural-gpu-smoke-v2/`（`corpus/manifest.json`、`model/metrics.json`、
`model/best.pt`、`model/final.pt`）；checkpoint 必须与 `corpus/tokenizer.json` 配套。

## 5. 尚未接通的接口与评估前置条件

- `audit-corpus` / `audit-scale-corpus` 仍只在源数据库查找动作规则，未安装并解析
  `gen_df_*`。对含生成式桥动作的 PA+ 语料，不能宣称这些 CLI 已完成全量认证；
  当前已有定义桥/有界实例的 DAG 重放测试和单例神经证书验证。
- `build-scale-corpus` 同样使用源规则模板解析，尚不能直接接续这种 PA+ 语料。
  `peano-htps generate` 也没有目录、有界实例和目标引导配置参数。
- 当前四策略评估中，只有构造 `TransformerPolicy` 的 neural/hybrid 分支会自动
  配置 PA+ 环境，uniform/heuristic 不会。正式 PA+ 对照之前，必须统一定义桥、
  闭项和候选顺序，再比较学习增益；旧数论库的历史对照不受此新增上下文影响。
- `--max-target-hints 0` 只去掉状态中的目标标签；环境排序仍使用元数据中的目标
  公式，不能当作“完全关闭目标引导”的消融。须分别控制生成权重、状态提示与
  推理枚举，并保留未参与引导的目标族作为 OOD 评估。

这些是当前代码限制，本次文档核对未修改实现。正式规模训练前应优先完成上述
统一回放、词表/环境一致性及无泄漏同预算评估。
