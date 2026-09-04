# 形式系统演进设计：从 PA+/Metamath 到类型化多后端

本文整理截至 2026-08-29 围绕形式系统表达力、定义机制、训练复杂度、
Equations/HTPS 以及 Lean 后端的讨论，作为后续原型和受控实验的实现参考。

2026-09-04 状态注记：本文保留原设计背景，其中 PA+ 分层定义的一部分已经实现：
68个高层关系、35个非逻辑目标、定义桥、有界实例、目标引导及基础神经管线。
分别见 [PA+ 定义库](pa-plus-definitions.md)、[随机生成](pa-plus-random-generation.md)
和[神经训练](pa-plus-neural-training.md)。统一 typed IR、通用原始递归/归纳类型
编译器和 Lean 多后端仍是计划，不能由现有关系宏推断为已经实现。

除上述已落地部分外，本文仍是设计方案。当前可信主线是项目已有的
PA+/Metamath 生成、搜索、证书编译和双重验证闭环；中间引理动作、连续潜在思维
和 HTPS 风格超图搜索的当前状态见
[增强 PA 下的 HTPS 数据、训练与推理](htps-design.md)。

## 1. 问题与目标

最初设计时 PA+ 系统的优势是内核小、替换语义明确、证书容易独立复验，并且已经形成
从认证数据生成到闭环求解的可运行系统。以下是当时的问题背景；新定义宏缓解了
表层重复编码，但并未消除底层 β 编码和一般数学对象表达限制：

1. 形式对象主要落在一阶自然数算术中，表达常用数学结构时层次过低；
2. 有限序列、递归轨迹、幂、计数函数和有理近似过度依赖 Gödel β 编码；
3. 一个数学上简单的定义或推理，可能展开成很长的关系和存在量词；
4. 模型容量被用于学习编码、解码和形式包装，而不完全是数学证明结构；
5. 继续扩展到列表、矩阵、多项式、函数和分析时，人工建设成本快速增加。

形式系统演进的目标不是追求最大的定理库，而是找到以下平衡：

- 定义简洁，但仍然是保守的、可机械检查的扩展；
- 模型看到的状态短、规则明确、候选动作有限；
- 能直接表示递归数据、代数结构和常用数学对象；
- 生成数据不需要自然语言，且每条样本都带可重放证明；
- 中间引理、连续潜在思维和 HTPS 可以跨后端复用；
- 最终是否迁移由同题、同预算的闭环实验决定，而不是仅凭表达美观决定。

## 2. 三种形式系统路线

### 2.1 继续扩展 PA+/Metamath

保留当前内核，以保守定义和新证明规则逐层扩展对象语言。

优势：

- 验证器和生成器已经可用；
- 状态、替换、作用域和 `$d` 条件完全可控；
- 动作空间确定，容易生成大量认证轨迹；
- 未压缩证明可以由官方 Metamath 实现独立复验；
- 适合研究搜索、引理发现和潜在思维本身。

限制：

- 一阶算术不是表达所有数学对象的经济中间语言；
- 递归对象通常需要关系化或编码化；
- 定义层变长会增加序列长度和证明深度；
- 扩展到高阶函数、结构和分析的工程成本较高。

这条路线适合作为可信基线和受限算术后端，不宜无边界地模拟整个现代数学库。

### 2.2 建立更强的自研 Equations+ 系统

不直接采用 `set.mm`，而是设计一个比 PA+ 丰富、比完整 Lean/Mathlib 更受限的
类型化形式系统。建议包含：

- 多排序一阶逻辑和等式；
- `Nat`、`Int`、`Rat`、`Bool`；
- 元组、有限列表、有限集合和有限映射；
- 半群、幺半群、群、半环、环、域和序结构；
- 多项式、有限和、有限积、向量和矩阵；
- 结构归纳、自然数归纳和受控的原始递归；
- 定义展开、重写、合取分解、存在引入等显式规则。

它可以显著减少 PA 编码负担，同时维持小内核和有限动作空间。但需要自行解决：

- 类型检查和多态实例化；
- 归纳类型及递归定义的可靠性；
- 定义的保守性检查；
- 规范化和证明项格式；
- 独立验证器以及长期兼容性。

如果完整实现这些能力，自研系统会逐渐接近一个小型证明助理。因此它更适合成为
统一的神经符号中间表示，而不是另造一个不断膨胀的最终基础系统。

### 2.3 使用 Lean 作为形式后端

Lean 的核心是带归纳类型的依赖类型论。函数、递归定义、列表、向量、结构和证明
项可以直接表达，elaborator 产生的核心表达式最终由小型可信 kernel 检查。Lean
官方文档对 elaboration、kernel checking 和核心类型论的边界有明确说明：
[Lean elaboration 与 kernel](https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/)。

Lean 的主要优势不是拥有更漂亮的表面语法，而是无需把所有对象先编码为自然数：

```text
PA+/β 编码：
  存在 B,C，使 beta(B,C,i,v) 表示递归轨迹的第 i 项为 v

Lean：
  def iterate : Nat → State → State
    | 0,     s => s
    | n + 1, s => step (iterate n s)
```

两者都可以严格验证，但后者保留了对象的类型和计算结构。

Lean 的代价也必须正视：

- elaboration、隐式参数、coercion 和 typeclass 增加环境复杂度；
- Mathlib 规模下的前提选择成为独立难题；
- 同一个目标可能存在大量等价 tactic 和实例化方式；
- `simp`、`ring`、`aesop` 等宏动作可能隐藏大量内部推理；
- kernel proof term 可能比表面 tactic script 大很多；
- 导入、命名空间和库版本会影响可复现性；
- 单次 tactic 执行和数据生成通常比当前轻量 Metamath 内核昂贵。

因此 Lean 不应直接替换当前主线，而应首先作为受限的实验后端。

## 3. 公理系统复杂度与训练成本

“公理系统更简单”不等于“模型所需训练量更小”。需要分别考察四类复杂度。

### 3.1 表达复杂度

同一数学命题在模型输入中需要多少节点或 token。底层编码越多：

- 状态序列越长；
- 有效上下文中数学结构所占比例越低；
- 模型需要额外学习编码的不变量；
- 搜索更容易把预算消耗在包装层。

这是当前 PA+ 使用 β 编码时最明显的成本。

### 3.2 证明复杂度

不能只比较最终 kernel proof 的原子步数，应同时记录：

- 数学抽象动作数；
- 模型实际预测的动作数；
- 展开后的 kernel/certificate 步数；
- 最大并行子目标数；
- 中间状态总 token 数；
- 中间引理被复用后节省的证书长度。

高层系统可能让抽象证明更短，但单步候选更多；低层系统可能单步简单，却需要更长
轨迹。最终训练和搜索成本由两者共同决定。

### 3.3 动作熵

在某个状态下，模型必须从多少合法候选中选择：

- Metamath 规则和替换通常较显式；
- Lean 中还要选择 theorem、tactic、位置、实例、隐式参数和局部表达式；
- 全量 Mathlib 会使 premise retrieval 成为必要模块。

LeanDojo 的结果表明，在大型 Lean 库中显式前提检索能够改善证明性能，而包含
未见前提的切分明显更困难：
[LeanDojo](https://arxiv.org/abs/2306.15626)。

### 3.4 环境执行成本

需要记录每秒可以完成的：

- 类型检查次数；
- 候选动作验证次数；
- 证明状态展开数；
- 完整证书重放数；
- 数据生成样本数。

更短的 Lean 状态不一定抵消 elaborator 和 tactic 执行的 CPU 成本。因此迁移决策
应使用 `solve rate / GPU-hour`、`solve rate / wall-clock-hour` 和生成成本，而不是
只比较 token accuracy。

## 4. 简洁而严格的定义机制

未来应把“数学定义”与“用于底层系统的编码实现”分离。建议采用四级定义机制。

### 4.1 纯语法宏

新记号在进入内核前完全展开到旧语言，不向逻辑增加新常量或新公理。

```text
x ≤ y  ↦  ∃z, x + z = y
```

优点是保守性最直接；缺点是完全展开后仍可能很长。模型侧可以保留宏节点，证书
编译时再展开，以避免训练序列反复承担相同包装。

### 4.2 已证明的定义性扩展

若要引入函数符号 `f`，先在旧系统中证明存在唯一性：

```text
∀x, ∃!y, R(x,y)
```

然后才允许引入满足 `R(x,f(x))` 的定义性常量。验证器必须记录该常量的定义来源，
并能在需要时消除定义。不能仅因一个符号“看起来像函数”就把它作为新公理加入。

### 4.3 受控递归定义

对自然数、列表等良基对象，定义器接受：

- 基础方程；
- 递归方程；
- 结构递减或良基证明；
- 由递归定理导出的存在唯一性证书。

在 Metamath 后端，它可以编译成已证明的递归关系或定义定理；在 Lean 后端，它可
编译成 recursor/结构递归定义并交给 kernel 检查。模型始终使用同一个高层定义
节点，不直接看到 β 编码的逐项细节。

### 4.4 不透明定理与宏 tactic

一个已验证定理可以作为后续证明的命名规则，但不能把未验证的计算过程当作公理。
自动化 tactic 必须产出可检查 proof term 或证书。

训练数据应同时保存：

- 宏动作，例如 `ring` 或规范化；
- 宏动作的显式参数和允许规则集；
- 展开后的细粒度证明 DAG；
- 最终 kernel proof/certificate。

否则模型可能只学会在大量问题上调用一个决策过程，而没有学到可组合的证明结构。

## 5. 统一类型化神经符号 IR

推荐把 Equations+ 定位为模型与形式后端之间的规范化 IR。模型不直接生成原始
Metamath 或 Lean 源码。

### 5.1 表达式

核心节点建议包括：

```text
Sort | Const | Local | App | Lam | Pi | Let
Eq | And | Or | Not | Imp | Forall | Exists
Constructor | Recursor | Match
```

第一版可以只实现一阶子集，后续再加入 `Lam`、`Pi` 和完整依赖类型能力。

规范化要求：

- 局部变量使用 De Bruijn index 或稳定 canonical ID；
- 后端常量映射到稳定的全限定 ID；
- 类型在 IR 中显式保存；
- 消除 notation、短名称和自然语言字符串；
- 规定 alpha-equivalence、定义展开和可交换节点的 canonicalization；
- 记录环境版本、规则包哈希和导入闭包。

### 5.2 动作

模型动作保持有限并结构化：

```text
APPLY(rule_id, substitution)
REWRITE(rule_id, position, direction)
INTRO(local_type)
EXACT(term)
CONSTRUCTOR(constructor_id)
INDUCT(local_id, recursor_id)
NORMALIZE(procedure_id, rule_pack_id)
PROPOSE_LEMMA(typed_expression)
```

符号环境先枚举或约束合法候选，模型主要负责排序。对必须生成的中间表达式，使用
类型约束解码器和增量检查，而不是自由生成 Lean tactic 文本。

### 5.3 状态和证书

统一状态包含：

```text
environment fingerprint
local context
ordered open goals
variable/type constraints
active definitions and rule pack
pending guarded lemmas
```

后端适配器负责：

```text
IR action
  → backend action
  → backend state transition
  → proof DAG edge
  → final kernel/certificate verification
```

连续潜在思维只影响 policy、critic、lemma gate 和 halt，不得修改形式状态，也不得
进入最终证书。

## 6. 无自然语言的 Lean 合成数据

“不使用自然语言”和“完全不使用人类形式库”需要分开讨论。

### 6.1 不使用自然语言，但使用形式规则库

可以从 Mathlib 或自选 Lean 模块中只抽取核心表达式和证明依赖，丢弃：

- 注释和 docstring；
- 人类语言题面；
- 漂移的短名称；
- 非必要的表面 notation；
- 源代码排版。

规则用稳定 ID 表示，模型输入为规范化类型化 AST。HTPS 已展示一条相近路线：
从 Mathlib 抽取兼容规则，在 Equations 环境中随机生成定理，再把定理和证明转换回
Lean。[HTPS 补充材料](https://papers.neurips.cc/paper_files/paper/2022/file/a8901c5e85fb8e1823bbf0f755053672-Supplemental-Conference.pdf)

### 6.2 不使用自然语言，也不使用人类证明

在受限领域中同样可行：

1. 人工指定类型、定义和可信起始规则；
2. 按类型生成项、变量、前提和边界条件；
3. 从已知事实正向应用认证规则，构造证明 DAG；
4. 以 DAG 根节点作为合成定理；
5. 反转 DAG，生成逐状态 policy、value 和 lemma 样本；
6. 用 Lean kernel 验证每个最终 proof term；
7. 按规范化证明骨架哈希切分数据集；
8. 以 HTPS/self-play 发现生成器未直接给出的新证明。

这种路线仍需要人决定“哪些数学对象和规则值得生成”，但不需要自然语言题库，也
不需要人工逐条书写证明。

### 6.3 Proof artifact 自监督

每个认证 proof term/DAG 可以进一步派生：

- 下一规则预测；
- 缺失子证明恢复；
- 下一中间引理预测；
- 前提选择；
- 子项类型预测；
- proof-state value；
- 证明骨架和具体实例的对比学习；
- 宏动作与细粒度动作之间的蒸馏。

PACT 证明了从 Lean kernel-level proof term 提取辅助任务能够缓解证明数据不足；其
实验中 held-out 证明成功率由 32% 提升到 48%。
[PACT](https://arxiv.org/abs/2102.06203)

### 6.4 难度和质量控制

随机良类型不等于有价值。生成器至少应过滤：

- 由单个恒真模板反复实例化的重复命题；
- 前提矛盾导致的平凡爆炸证明；
- 未使用前提和可删除变量；
- 归一化后相同的命题；
- 单一万能 tactic 一步解决的大量同质样本；
- 证明骨架与训练集重复的测试题；
- 过短、过长或搜索预算之外的轨迹。

建议按“随机/启发式求解器所需展开数”动态估计难度，而不是只用参考证明深度。

## 7. 面向常用定理的领域扩展顺序

不建议一次实现“多数数学”。应按依赖层逐步建立领域包。

### L0：逻辑、等式和计算基础

- 命题逻辑、量词、等式和替换；
- 自然数、整数、有理数；
- 递归、归纳、有限列表和元组；
- 规范化和小规模决策过程。

### L1：通用代数

- 半群、幺半群、群；
- 半环、环、交换环、域；
- 序、格、绝对值；
- 同态、子结构和有限生成结构。

### L2：离散数学

- 有限集合、有限和与有限积；
- 组合恒等式、二项式系数；
- 整除、同余、素数和初等数论；
- 图、路径和有限计数。

### L3：符号代数和线性代数

- 一元/多元多项式；
- 向量、矩阵、线性映射；
- 行列式、秩的有限维子集；
- 可验证的归一化证书。

### L4：实数与基础分析

- 实数序和完备性接口；
- 序列、极限、连续性；
- 有限求和估计和初等函数；
- 微积分和测度论放在较晚阶段。

若采用自研 Equations+，建议止于 L2 或 L3 的受限子集。L4 更适合通过 Lean 后端
获得成熟定义和 kernel 检查，而不是在 PA 中继续堆叠编码。

## 8. 中间引理和连续潜在思维的迁移

### 8.1 中间引理

形式语义保持当前 guarded AND edge：

```text
Γ ⊢ G
  -- PROPOSE_LEMMA L -->
    1. Γ ⊢ L
    2. Γ, L ⊢ G
```

Lean 后端还必须检查：

- `L : Prop` 在当前局部上下文中良类型；
- `L` 不含逃逸 metavariable 或无效局部变量；
- 第一分支产生的 proof term 类型确为 `L`；
- 第二分支只能在第一分支成功后使用该局部事实；
- 最终证书将局部引理构造成 `have`/lambda application 或内联 proof term。

引理训练正样本优先来自证明 DAG 的高价值 cut：

- 被多个后续分支复用；
- 显著减少剩余搜索展开；
- 将一个高分支问题分为两个较低分支问题；
- 不是当前目标或已有前提的改名；
- 在不同具体实例中共享相同规范化骨架。

### 8.2 连续潜在思维

现有 recurrent latent state、halt head 和 ponder cost 可以独立于后端。需要增加的
训练信号包括：

- 下一动作与候选排序损失；
- 状态/超边价值损失；
- 是否提出引理的 gate 损失；
- 引理表达式的类型约束生成损失；
- 搜索预算或剩余证明成本预测；
- halt 的 ponder 正则。

潜在向量不作为证明事实。任何由 latent state 建议的规则、替换或引理，都必须先
通过后端类型检查并进入显式搜索状态。

## 9. 推荐架构

```text
                   canonical typed proof IR
                 /                          \
        Metamath PA+ adapter          Lean adapter
        - 极简可信基线                - 丰富类型与递归
        - 快速生成/复验               - 受限 Mathlib 规则包
                 \                          /
             candidate enumeration / checking
                              |
              policy + critic + lemma generator
                    + continuous latent thought
                              |
                   HTPS AND/OR hypergraph
                              |
             backend certificate / kernel checking
```

这允许在不丢弃当前成果的情况下回答两个关键研究问题：

1. 性能提升来自更好的形式表达，还是来自不同的数据和规则分布？
2. Lean 减少的表达/证明长度，能否抵消更高的动作熵和执行成本？

## 10. 分阶段实施计划

为避免与现有 P0–P3 路线混淆，本节使用 `FS` 编号。

### FS0：度量当前编码成本

不改内核，选择 100–300 个代表性 theorem family，记录：

- 表达式 AST 节点和 token 数；
- 抽象证明步和展开证书步；
- β 编码相关节点占比；
- 生成、搜索和复验耗时；
- 模型在包装步骤与数学步骤上的错误分布。

交付物是可重复运行的形式系统成本基线。

### FS1：定义层和 IR 设计

- 定义 canonical typed AST；
- 定义结构化动作 schema；
- 明确 alpha/definition/normal-form 等价；
- 建立稳定常量 ID、环境指纹和版本格式；
- 将当前 Metamath 状态无损映射到 IR；
- 保持旧 tokenizer 和 checkpoint 不变，使用新文件和显式版本号。

验收条件是 IR 往返 Metamath 后，现有证书语义不变。

### FS2：简洁定义原型

先实现以下三类高收益对象：

- 原始递归函数；
- 有限列表/递归轨迹；
- 有限和、多项式和矩阵的受限子集。

要求模型侧不再展开 β 编码，证书侧仍可产生当前 Metamath 内核接受的证明。对不可
经济编译的对象要明确拒绝，不用新公理掩盖实现困难。

### FS3：受限 Lean-Equations+ 后端

第一版规则包建议限制为 500–2000 条，领域包括：

- `Nat`、`Int`、`Rat`；
- 等式、序和初等逻辑；
- 列表、有限和与多项式；
- 小规模矩阵；
- 自然数和结构归纳。

目标不是证明整个 Mathlib，而是生成 100,000–300,000 条 kernel-checked 样本，
覆盖约 5–50 个抽象动作的证明，验证数据质量、吞吐和搜索接口。

### FS4：同题同预算对照

将同一批 theorem family 编译到两个后端，固定：

- 模型参数量和优化器；
- 训练有效 token 或 GPU 时长；
- 搜索 wall time、节点预算和候选上限；
- 数据 split 的证明骨架隔离规则；
- 多个固定随机 seed。

主要指标：

| 指标 | 目的 |
| --- | --- |
| 平均状态 token/AST 节点 | 测量表达开销 |
| 抽象动作数和证书步数 | 区分模型难度与内核展开 |
| 平均合法候选数 | 测量动作熵 |
| 数据生成样本/CPU-hour | 测量数据成本 |
| search expansions/solved | 测量搜索效率 |
| solve rate/GPU-hour | 测量训练收益 |
| solve rate/wall-clock-hour | 纳入环境执行成本 |
| OOD proof-skeleton solve rate | 测量组合泛化 |
| kernel/外部验证通过率 | 保持可信边界 |

### FS5：迁移决策

建议把下列阈值作为研究门槛，而不是绝对承诺：

- Lean/新 IR 将模型可见状态或抽象证明长度至少降低约 2 倍；
- 最终 kernel 检查通过率为 100%；
- `solve rate/GPU-hour` 至少提升约 25%；
- 数据生成 CPU 成本不超过 Metamath 对照约 2 倍，或能通过缓存摊销；
- 在 proof-skeleton 隔离测试上没有依靠模板泄漏取得虚假增益。

若通过门槛，新增数学领域优先进入 Lean 后端；若未通过，继续保留 PA+/Metamath
主线，并只把 IR 用于压缩定义和改进模型表示。

## 11. 第一轮实验建议

### 11.1 配对题族

首轮不要选择依赖大量 Mathlib 自动化的题目。建议使用：

- 加法、乘法、序的代数恒等式；
- 整除和同余；
- 有限和与简单递推；
- 列表长度、拼接和映射；
- 二项式低阶实例；
- 2×2、3×3 矩阵恒等式；
- 需要一个中间引理的组合题。

每个题族生成保持变量、常量和类型变化的实例，并按抽象证明骨架隔离 split。

### 11.2 模型规模

先使用当前约 100M 参数级别模型，不因切换后端同时改变模型规模。这样才能识别
收益来自形式系统、数据表示还是单纯增加参数。

### 11.3 消融

至少比较：

1. 原始 Metamath token；
2. Metamath 后端 + canonical typed IR；
3. Lean 后端 + 同一 IR；
4. Lean 原始 pretty-printed tactic state；
5. 无引理动作；
6. 有 guarded lemma；
7. 无 latent recurrence；
8. 有 latent recurrence；
9. 宏 tactic；
10. 展开的细粒度动作。

其中 2 对 1 测量表示压缩收益，3 对 2 测量后端表达力收益，4 对 3 测量规范化 IR
是否优于直接学习 Lean 文本。

## 12. 风险和防护

### 12.1 定义偷渡公理

任何新函数或对象必须是纯宏、已有项的缩写，或附带存在唯一性/递归合法性证明。
不能把难以定义的对象直接声明为新常量并附加期望性质。

### 12.2 训练/测试泄漏

除最终命题外，还必须对以下内容做去重或隔离：

- 规范化证明骨架；
- 定义展开后的目标；
- 交换/结合/变量改名等价类；
- 合成 seed family；
- 直接父定理和中间引理模板。

### 12.3 自动化掩盖推理深度

宏 tactic 的成功率不能单独作为证明能力。报告中必须同时给出展开证明的依赖数、
证书大小以及在禁用该宏 tactic 后的对照。

### 12.4 库规模失控

Lean 后端使用显式版本化的 rule pack，不默认暴露整个 Mathlib。每条规则记录来源、
完整名称、类型、依赖和哈希。模型只能选择当前证明环境中可访问的规则。

### 12.5 验证器耦合

最终成功必须重新启动干净环境，从锁定依赖编译并由 Lean kernel 检查；Metamath
后端继续进行项目内和官方实现双验证。缓存和模型输出不能替代最终重放。

## 13. 当前建议

1. 保留 PA+/Metamath 作为可信基线，不立即重写当前闭环。
2. 停止在模型可见层继续增加长 β 编码；优先设计 typed IR 和定义宏。
3. 将 Equations+ 设计为统一神经符号 IR，而不是新的大型最终基础系统。
4. 建立受限 Lean-Equations+ 后端，只使用规范化形式表达，不引入自然语言。
5. 复用现有候选策略、critic、guarded lemma、连续潜在思维和 HTPS。
6. 用同题同预算实验决定哪些领域迁移到 Lean，而不是一次性替换。
7. 即使采用 Lean，也不让模型自由拼写原始 Lean 源码；优先预测有限、类型化、可即时
   检查的结构化动作。

简要判断是：

- 小型 PA、验证器研究和极致可审计性：Metamath 更合适；
- 递归结构、代数、矩阵和常用高层数学：Lean 更有优势；
- 当前项目最稳妥的长期形态：统一 typed IR 下的 Metamath 与 Lean 双后端。

## 14. 参考资料

- Lample et al., [HyperTree Proof Search for Neural Theorem Proving](https://papers.neurips.cc/paper_files/paper/2022/file/a8901c5e85fb8e1823bbf0f755053672-Paper-Conference.pdf)
- Lample et al., [HTPS supplementary material](https://papers.neurips.cc/paper_files/paper/2022/file/a8901c5e85fb8e1823bbf0f755053672-Supplemental-Conference.pdf)
- Han et al., [Proof Artifact Co-training for Theorem Proving with Language Models](https://arxiv.org/abs/2102.06203)
- Yang et al., [LeanDojo: Theorem Proving with Retrieval-Augmented Language Models](https://arxiv.org/abs/2306.15626)
- Lean, [Elaboration and Compilation](https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/)
- Lean, [The Type System](https://lean-lang.org/doc/reference/latest/The-Type-System/)
- Mathlib, [module documentation](https://leanprover-community.github.io/mathlib4_docs/Mathlib.html)
