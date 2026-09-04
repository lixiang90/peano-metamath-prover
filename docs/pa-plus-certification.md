# PA+ 认证与公平评估补齐

2026-09-04，本轮只完成认证、评估一致性和回归验收，没有新增公理、规模训练或
宣称证明能力提升。HTPS 生成器与 PA+ 的下一步数据集成不在本轮范围内。

## 1. 统一动作解析与确定的变量槽位

`parse_tactic_tokens(..., environment=...)` 从已配置环境查找规则，不再局限于源库。
`audit-corpus` 和 `audit-scale-corpus` 使用同一入口。环境机械生成定义桥后，将其
扁平证书交给项目验证器再安装；桥名和其引用的源规则若被评估排除，则桥不能加入。
原始数据库不会因安装宏而增加可信公理。

本轮首次跨进程回放旧616例语料时，116条有界实例动作失败。原因是生成规则没有
源库 `$f` 顺序时，旧实现按来自集合的 `variable_types` 顺序分配类型化槽位；
不同进程可能把同一槽位解释成不同变量。源证明 DAG 的可靠性不自动保证序列化
训练动作的可靠性。

新语料统一按变量名排序定义桥的槽位，tokenizer 元数据记录
`bridge_variable_order: sorted-v1`。没有该标识的旧 PA+ tokenizer 读为
`legacy-unordered`：普通源规则仍兼容，但旧桥动作审计明确失败，要求重新生成。
不通过重合一猜测、丢弃绑定或静默改写历史样本来冒充原动作回放成功。

历史 checkpoint 不会被改写；可以保留其权重作后续初始化，但旧 PA+ 动作基线须
重新生成、审计和训练。2026-08-31 的指标是历史工程记录，不是新认证训练基线。

## 2. 公平环境与三层引导

普通评估和四策略 MCTS 都先配置环境，再选择 uniform/heuristic/neural/hybrid。
神经策略不再重置评估者已经选择的引导设置。每个 attempt 使用新环境，并报告
`environment.fingerprint`、定义桥数、自然数上界、引导状态和排除规则；同题各
策略应具有相同指纹。这个指纹核对的是候选环境，不表示各策略会访问相同节点。

| 层次 | 关闭方法 | 保留的能力 |
| --- | --- | --- |
| 随机生成目标引导 | `build-corpus --target-guidance-weight 0` | 合法组合、定义覆盖采样与配额 |
| 训练状态目标标签 | `build-corpus --max-target-hints 0` | 原目标/前提、定义符号、类型与 `$d` |
| 推理模型目标标签 | `--no-model-target-hints` | 原目标/前提与词表 ID；不改 tokenizer 文件 |
| 推理目标排序 | `--no-inference-target-guidance` | 相同定义桥、同一有界项集合 |

关闭生成权重不等于去掉模型提示；关闭标签也不等于取消推理排序。完整无目标引导
对照需要在相应阶段全部关闭。即使指标达到35/35触达，也不是35个目标被证明。

## 3. HTPS 的统一证书门槛

`peano-htps evaluate`、`collect` 和 `closed-loop` 支持：

- `--external-verifier /path/to/metamath`；
- `--require-external-verification`；
- `--external-timeout-seconds 60`；
- 上述模型标签和推理目标排序开关。

项目内证明重放始终必需。要求外部验证时，程序不存在、失败、超时、异常或输出
不明确均不计为 certified；已解但外部未通过的路径不进入成功 replay。
明确给定的无效程序路径不会悄悄回退到另一程序。结果保留验证状态、程序哈希和
`internal_only` / `internal+external` 级别，以便区分口径。

```bash
python -m htps_prover evaluate MODEL.pt TOKENIZER.json POLICY_TEST.jsonl \
  formal/peano-pa-plus.mm outputs/htps-certified.json \
  --device cpu --limit 16 --simulations 40 --expansions 100 \
  --external-verifier /path/to/metamath --require-external-verification \
  --external-timeout-seconds 60 \
  --no-model-target-hints --no-inference-target-guidance
```

该命令要求配套的 latent checkpoint、tokenizer 和 HTPS policy 数据；不表示 PA+
`build-corpus` 数据已经可以直接代替 HTPS forward 数据。

## 4. 本地验收与复现

新建目录保存修复后的语料，保留旧训练产物不变。在仓库根目录执行以下 Bash
命令；已有 PyTorch 的 PowerShell 可先设 `$env:PYTHONPATH='src'`，把续行改为反引号：

```bash
python -m neural_prover build-corpus formal/peano-pa-plus.mm \
  outputs/pa-plus-certification-v1/corpus \
  --seeds 7,11 --steps-per-seed 300 \
  --max-proof-depth 8 --max-ast-depth 64 --max-variables 24 \
  --definition-catalog formal/pa-plus-definitions.json \
  --bootstrap-definitions --definition-coverage-weight 3 \
  --bounded-nat-max 2 --ground-instances-per-predicate 1 \
  --target-guidance-weight 4 --max-definition-only-search-per-predicate 1 \
  --max-state-tokens 1024 --max-action-tokens 512

python -m neural_prover audit-corpus formal/peano-pa-plus.mm \
  outputs/pa-plus-certification-v1/corpus \
  --output outputs/pa-plus-certification-v1/audit.json

python -m unittest discover -s tests -p test_certification.py -v
```

本地验收覆盖：

- 新语料521/58/37，共616条动作全量回放通过；报告在上述忽略目录的 `audit.json`。
- 399条基础/桥/实例语料在另一 `PYTHONHASHSEED=193` 的进程中回放通过；136条
  有界实例转换成测试 gzip 分片后全部审计通过。
- 启用/关闭推理引导各跑一次四策略小预算测试，同一设置下四策略环境指纹一致。
  该测试用于配置一致性，不用于统计学习策略的胜率。
- `positive`、`gcdrel`、`qadd` 三个定义桥证书通过项目内核和本地官方 Metamath；
  HTPS 单步 sanity 目标也通过强制外部门槛。
- 模拟外部失败/缺失/异常时拒绝 certified；HTPS 已解路径外部失败时不写成功 replay。
- 模型标签与推理排序互不覆盖；重配置清缓存；排除源规则不能借助桥绕过。

官方 Metamath 测试优先使用 `METAMATH_EXECUTABLE`，否则使用本地
`outputs/tools/metamath/metamath.exe`。两者均未配置时会跳过专门的真实外部桥测试，
不能把跳过等同于已完成外部验证。显式设置的程序无效时测试失败，不静默跳过。
统一主线 CI 会构建固定版本的官方验证器，通过该环境变量运行 PA+ 桥与 HTPS 集成测试。

后续仍需：PA+ → HTPS 数据贯通、Scale 模板集成、升级 checkpoint 微调入口、
未参与目标引导的保留集，以及多种子同预算的正式 solve-rate 实验。
