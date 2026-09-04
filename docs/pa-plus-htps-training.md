# PA+ → 数据与训练贯通

2026-09-04。本轮贯通共享 PA+ 生成器与 HTPS forward 数据、动作审计、latent 监督
训练，并增加基础 checkpoint 微调入口。没有增加公理，也没有完成 Scale 桥模板
扩增或正式 solve-rate 对照。全局入口见[训练与评估指南](training-and-evaluation.md)。

## 1. 数据契约

`htps_prover generate` 现支持与 `neural_prover build-corpus` 一致的目录、定义桥、
有界项、35目标引导与包装限流参数。编码生成式规则时从生成库查找规则，不再只查
原始 `.mm` 标签；桥仍只是可证明的搜索宏。

policy 和 lemma 记录包含 `generation_kind`、`guidance_target`、`definition_support`、
词表指纹和证明骨架。构建时回放每个将要输出的动作，lemma 还必须通过真实 cut
调度器的类型、大小和“不能重复目标/已有假设”等检查。无法编码或不合适的候选
会被过滤并报告计数；这与签发未通过验证的训练动作不同。

分区按“相同递归骨架或相同根状态”的连通组确定，policy/lemma 属于同一组，跨种子
的相同 state/action 去重。它不能替代定义族 OOD 或祖先依赖泄漏审计。新数据格式
为 `peano-htps-forward-dag-v2`，旧 v1 无需原地覆盖，PA+ 旧桥动作仍须重新生成。

生成器保存 `action-audit.json`；`audit-data` 可在另一个进程重放全部序列化动作并
检查分区冲突，发现失败时 CLI 返回非零。完整证明 DAG 的深依赖族导出重放另外
统计；动作合法、完整证明可回放、外部认证是三个不同层次，不能互相代替。

## 2. PA+ HTPS 从头训练

在仓库根目录执行以下 Bash 命令；PowerShell 将续行改成反引号，已有 PyTorch 时
可先设 `$env:PYTHONPATH='src'`。CPU 小模型即可检查接口；GPU 实验改 `--device cuda`。

```bash
python -m htps_prover generate formal/peano-pa-plus.mm outputs/pa-htps/data \
  --steps 300 --seeds 7,11 --max-proof-depth 8 --max-ast-depth 64 \
  --max-variables 24 --max-state-tokens 1024 --max-action-tokens 512 \
  --definition-catalog formal/pa-plus-definitions.json \
  --bootstrap-definitions --definition-coverage-weight 3 \
  --bounded-nat-max 2 --ground-instances-per-predicate 1 \
  --target-guidance-weight 4 --max-definition-only-search-per-predicate 1

python -m htps_prover audit-data formal/peano-pa-plus.mm outputs/pa-htps/data \
  --output outputs/pa-htps/audit.json

python -m htps_prover init-model outputs/pa-htps/data/tokenizer.json \
  outputs/pa-htps/initial.pt --d-model 32 --heads 4 \
  --encoder-layers 1 --decoder-layers 1 --ffn 128 \
  --max-state-tokens 1024 --max-action-tokens 512 --thought-steps 2

python -m htps_prover train-supervised outputs/pa-htps/initial.pt \
  outputs/pa-htps/data/tokenizer.json outputs/pa-htps/data/policy_train.jsonl \
  outputs/pa-htps/sft.pt --lemmas outputs/pa-htps/data/lemma_train.jsonl \
  --device cpu --epochs 2 --batch-size 8 --learning-rate 0.0005 \
  --candidate-loss-weight 0.25

python -m htps_prover evaluate outputs/pa-htps/sft.pt \
  outputs/pa-htps/data/tokenizer.json outputs/pa-htps/data/policy_test.jsonl \
  formal/peano-pa-plus.mm outputs/pa-htps/evaluation.json \
  --device cpu --limit 4 --simulations 8 --expansions 16 --branching 24 \
  --external-verifier /path/to/metamath --require-external-verification
```

把验证器路径替换为本机程序；限4条仅用于接线检查，不是有代表性的 held-out 解题率。
同步 `closed-loop` / `collect` 可使用这里的配套 SFT checkpoint 与 `policy_train.jsonl`；
后者不应换成测试分区。命令详见 [HTPS 设计](htps-design.md)。

SFT 默认对 PA+ 开启按类型加权采样：source=0.5、definition_bridge=0.75、
bounded_instance=3、search=1；lemma 保留其根证明的生成类型。可用
`--no-pa-plus-balanced-sampling` 关闭。每轮抽取同样数量样本，但有放回采样不保证
每条记录都被访问。报告实际抽到的各类数量，不把库大小等同于消费量。

`--candidate-loss-weight` 默认0；正值启用批内多正例损失，重复动作不是负例，
但其他合法动作仍可能被视为负例。它只是检索式训练，不是合法动作集上的策略
准确率。latent 状态通过动作/价值和 ponder 梯度学习，不引入自然语言思维链。

## 3. 从旧 checkpoint 微调

基础模型路径：先扩展词表和权重，再让生成器沿用扩展后的 tokenizer，最后显式
加载配套 checkpoint。输出使用新目录，旧文件不覆盖。例中上下文1024/512要求
旧 checkpoint 至少支持这个长度，否则应降低生成上限。

```bash
python -m neural_prover upgrade-pa-plus OLD.pt OLD-tokenizer.json \
  formal/peano-pa-plus.mm formal/pa-plus-definitions.json \
  outputs/pa-finetune/upgraded.pt outputs/pa-finetune/tokenizer.json \
  --bounded-nat-max 2 --max-target-hints 3

python -m neural_prover build-corpus formal/peano-pa-plus.mm \
  outputs/pa-finetune/corpus --base-tokenizer outputs/pa-finetune/tokenizer.json \
  --seeds 7,11 --steps-per-seed 300 --max-proof-depth 8 \
  --max-ast-depth 64 --max-variables 24 --max-state-tokens 1024 --max-action-tokens 512 \
  --definition-catalog formal/pa-plus-definitions.json --bootstrap-definitions \
  --bounded-nat-max 2 --ground-instances-per-predicate 1 \
  --target-guidance-weight 4 --max-definition-only-search-per-predicate 1

python -m neural_prover audit-corpus formal/peano-pa-plus.mm \
  outputs/pa-finetune/corpus --output outputs/pa-finetune/audit.json

python -m neural_prover train outputs/pa-finetune/corpus outputs/pa-finetune/model \
  --checkpoint outputs/pa-finetune/upgraded.pt \
  --checkpoint-tokenizer outputs/pa-finetune/tokenizer.json \
  --device cpu --epochs 1 --batch-size 8 --learning-rate 0.00002
```

加载模型时使用 checkpoint 自带结构，忽略新建模型的宽度/层数默认值；优化器重新
初始化，不宣称精确恢复旧训练轨迹。普通 `train` 只接收基础模型；latent checkpoint
用 `htps_prover train-supervised`。从基础模型转 HTPS 时，按“PA+ 升级 → `init-latent`
→ `generate --base-tokenizer`（使用最终 latent tokenizer）→ latent SFT”执行。

生成器只追加新 token，保留已有 ID。数据训练入口检查 token 与 ID 一致；新 checkpoint
校验包含 token 顺序和 PA+ 上下文的指纹，同样大小但排序不同也会被拒绝。旧 checkpoint
没有指纹时，只能依赖用户显式提供正确的配套 tokenizer，不能追溯证明原 ID 对齐。
`legacy-unordered` PA+ 数据被拒绝，不可用重新贴标签代替重建。

## 4. 本地验收与边界

本轮使用 `outputs/pa-htps-integration-v1/`（被 Git 忽略），CPU 限制为2个 PyTorch
线程，未启动远程任务或 GPU 训练：

- 两个种子各300步，480条 policy（136桥、136有界实例、208搜索）和612条 lemma；
  train=378/483、validation=55/70、test=47/59（各对为 policy/lemma）。
- 1,092条输出动作全量回放通过；另有8个深证明依赖族导出、重新解析，重放46条
  生成声明。40个不适合真实 cut 的候选被过滤；15条 policy 编码失败被计数，不纳入训练。
- 32维、单层 encoder/decoder、2步潜在预算，在861条训练记录上有放回采样训练2轮。
  训练总损失6.0057→4.0021，候选损失2.0637→1.8443。这是训练损失，不是验证集成绩。
- 用该 SFT checkpoint 对测试分区最前4条定义桥目标做 HTPS 接线验证，4/4通过项目内
  与官方 Metamath 双验证，证书长度为33/26/26/82个标签。都是浅定义目标，不能当作
  通用保留集胜率，更不能归因于神经学习的增益。
- 新回归覆盖另一个 `PYTHONHASHSEED` 的全量回放、跨种子去重/分组、SFT 的桥/实例/
  引理消费、等大小错误词表拒绝、token/ID 篡改拒绝，以及基础模型升级后微调且保留旧文件。
  训练后还针对一个短有界实例运行 HTPS 并验证证书；配置了外部验证器时强制外部
  通过，CI 的官方验证器任务包含此路径。完整本地回归共98项通过。

当前数据仍以定义桥与浅搜索为主；35目标引导触达不等于35目标已证。定义族 OOD、
跨后端公平预算、多种子能力对照及完整 Scale 模板集成仍是后续工作。
