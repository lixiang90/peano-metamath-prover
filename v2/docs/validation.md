# V2 本地验收记录

日期：2026-10-03。环境：Windows、Python 3.12.5、PyTorch 2.10.0+cu130。所有本轮开发训练/测试使用 CPU，`CUDA_VISIBLE_DEVICES=-1`，`OMP_NUM_THREADS=MKL_NUM_THREADS=1`；原有 3090 训练保持运行。该记录是工程闭环验收，不是模型能力基准报告。

## 回归与形式认证

全套执行：

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONPATH='src;v2'
$env:METAMATH_EXECUTABLE=(Join-Path (Get-Location) 'outputs/tools/metamath/metamath.exe')
python -c "import torch,pytest; torch.set_num_threads(1); raise SystemExit(pytest.main(['-q','--tb=short']))"
```

最终全套 **252 passed**（56.19 秒）；唯一 warning 来自 v1 PyTorch nested-tensor API。v1 独立复跑 **168 passed**。v2 覆盖双因果 decoder/RoPE、1/2/4 引理槽、顺序与梯度、无文本旁路、NTP 全 token 计数、SFT 失败掩码、真实 RLVR 梯度、库预算与陈旧索引、条件引理与 `$d`、回退作用域、上下文与证明展开硬预算以及官方验证。

数据/逻辑边界审查额外关闭了以下问题：

- 辅助变量和假设重排/重复不能把同一目标伪装成另一条库定理或另一个 split。
- 必需变量及 `$d` 条件不能藏进 proof-only 元数据来改变目标。
- 主模型默认不再从 SEARCH/READ 文本直接读取完整引理，避免绕过引理编码器。
- NTP 的初始头部没有未来事实；分块预测每个正文目标恰好一次。
- FINISH 后的未执行动作不能冒充已重放教师动作。
- 随机生成预算在多个种子之间分配，防止第一个种子的定义桥占满全部配额。
- 引理宏的重复前提替换在分配展开列表前检查证明长度预算；超限回滚并报告 unknown，防止嵌套引用造成指数级分配。

## 随机语料与官方验证

固定 `PYTHONHASHSEED=0` 后执行：

```powershell
python -m pa_prover_v2 generate formal/peano-pa-plus.mm outputs/v2-final/corpus --seeds 7,11,19 --steps-per-seed 50 --max-examples 48 --library-items 12 --axiom-instances-per-seed 6
python -m pa_prover_v2 audit formal/peano-pa-plus.mm outputs/v2-final/corpus --require-external --output outputs/v2-final/audit.json
```

| 项目 | 结果 |
|---|---:|
| 认证教师 episode | 48 |
| train / validation / test | 37 / 6 / 5 |
| 随机一步规则实例 / 组合 DAG | 18 / 30 |
| 每个种子生成数 | 16 / 16 / 16 |
| 去重过滤 | 3 |
| 官方 Metamath 认证 | **48 / 48** |
| 初始定理库 | 12 条，受 2,000,000 字节预算约束 |
| 实际草稿 USE / 全局库 USE | 30 / 1 |

第二个独立 Python 进程以相同配置重建到 `outputs/v2-final/repro-corpus`，全部 manifest、数据分片与定理库文件哈希一致。需要同时固定 Python hash seed 与采样 seed；只给采样 seed 不足以保证 v1 共享生成器逐字节稳定。

当前理论指纹：`c8d9514c37173545bbfe2ba1539d255ef6c1ff085738ed45c06814036f34c06d`。新数据格式为 3、库格式为 `pa-prover-v2-library-v2`；开发期间较早生成物被拒绝加载，应重建，不应静默迁移分组身份。

## NTP 与 Agent SFT

CPU smoke 使用 `d_model=32`、1 层主 decoder、1 层引理 decoder、4 heads、2 summary slots、16,384 token 上限，batch size 1。这是小型连接验证模型，不是推荐规模配置。

```powershell
python -m pa_prover_v2 pretrain formal/peano-pa-plus.mm outputs/v2-final/corpus outputs/v2-final/ntp --d-model 32 --layers 1 --lemma-layers 1 --heads 4 --context 16384 --steps 12 --batch-size 1
python -m pa_prover_v2 sft formal/peano-pa-plus.mm outputs/v2-final/corpus outputs/v2-final/sft --checkpoint outputs/v2-final/ntp/model.pt --steps 16 --batch-size 1
```

| 训练 | 首步 loss | 末步 loss | 引理有梯度步数 |
|---|---:|---:|---:|
| NTP，12 步 | 5.562800 | 5.432281 | 8 |
| SFT，16 步 | 5.431151 | 5.214544 | 6 |

NTP 保留全部 338,457 个正文目标；SFT 保留 306 个动作样本，12 个完整上下文超限样本明确过滤。SFT 不监督 37 次受控失败动作，但后续上下文保留这些尝试和反馈。小样本 loss 下降只说明可训练性，不能证明泛化能力改善。

## RLVR 与神经搜索

```powershell
python -m pa_prover_v2 rlvr formal/peano-pa-plus.mm outputs/v2-final/corpus outputs/v2-final/sft/model.pt outputs/v2-final/rlvr --groups 4 --group-size 4 --max-steps 4 --max-candidates 2 --require-external
python -m pa_prover_v2 evaluate formal/peano-pa-plus.mm outputs/v2-final/corpus outputs/v2-final/sft/model.pt outputs/v2-final/eval --limit 5 --samples 2 --max-steps 6 --max-candidates 4 --neural-retrieval --require-external
```

RLVR 的 16 次 rollout 中，9 次通过官方验证；4 组中有 2 组产生实际策略梯度更新，另 2 组分别全成功、全失败，因相对优势为零而跳过。两个更新组的奖励分别为 `[1, 1, 1, 0]` 和 `[1, 0, 1, 0]`。

独立 test split 的 5 个目标中，SFT 检查点以每题最多 2 次采样、每次最多 6 步证明了 3 个，全部要求官方验证；启用了冻结的引理向量检索。这里只验证推理闭环，样本包含一步课程题且数量极少，不能视为困难题基准或 RLVR 相对 SFT 的改善证据。该评估使用 SFT 检查点，而非 RLVR 更新后的检查点。

单元回归另用实际 PA+ 内核构造同一目标的成功/失败轨迹，验证奖励 `[1, 0]`、更新后正确证明动作概率增加、全零优势组不修改参数、伪造 success 和错误目标不获奖励。探索混合、长度归一化、候选集与 old log-prob 在 rollout/update 中一致。

## 包与旧版本

旧生成器 `python -m metamath_generator formal/peano.mm --steps 100 --output-dir outputs/v2-validation/v1-generator-smoke` 正常，产生 29 个证明对象。

`python -m build --no-isolation --outdir outputs/v2-final/dist` 成功构建 wheel 和 sdist；检查归档中四个包、PA+ 文件、v2 文档和 CLI 入口，并把 wheel 以 `--no-deps --no-index --target` 安装到独立目录。离开源码搜索路径后，四包导入均指向该目录，`python -m pa_prover_v2 --help` 正常。

发布包含 `pa_prover_v2`、`pa-prover-v2` CLI、PA+ 文件、原三个包和全部 v2 文档。模型、数据、官方验证器二进制及运行日志位于被忽略的 `outputs/`，不推送到源码仓库。GitHub 推送结果以最终提交和远端分支为准。

## 剩余研究问题

尚未验证检索向量优于词法检索、库机制提升困难题成功率或 RLVR 在深证明上稳定优于 SFT。当前只有一次全局库复用的随机语料命中，后续应增加共享中间结构的随机组合比例并做等预算消融。首版没有递归引理编码、latent CoT、KV cache 或长轨迹分页；循环计算的研究依据与分阶段实验见 [引理依赖与循环计算](lemma-dependencies.md)。
