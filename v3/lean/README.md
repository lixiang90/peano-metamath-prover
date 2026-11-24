# 用 Lean 承载合成数学

v3 的主实验沿用 PA+/Metamath。本目录考察用 Lean 4 的依赖类型论表达数学对象，用项目自己生成并验证的引理启动搜索。同一个算术命题可以在两边陈述，下面分别记录命题翻译、具体证明和数据生成的验证结果。[v3 总览](../README.md)

想先动手造题，可以直接运行随机定理生成器。熟悉 Metamath 的读者也可以先看一个 32 条语句的小数据库。把依赖函数用于自然数 `0` 或 `1` 时，结果类型分别是对应长度的向量。三个 `$p` 声明均可用官方 Metamath 的 `VERIFY PROOF *` 检查。[随机定理生成示例](#从证明生成训练题) · [依赖函数示例](dependent-pi.mm)

## 重大命题可以怎样写

费马大定理可以写成自然数和乘方的命题。黎曼假设也有算术等价形式，例如某个丢番图方程无解。给 PA 加保守定义，可以让这样的陈述易读，同时保持原语言中的可证结论。把命题写成算术、证明两种陈述等价、找到命题的证明，是三项各自需要论证的工作。特别是 PA 的一阶变量取值为自然数，不能直接量化任意复函数。[Matiyasevich 对黎曼假设算术等价形式的说明](https://logic.pdmi.ras.ru/~yumat/publications/publications.php?ilang=eng&imykey=144&istate=state_show_paper) · [本项目 PA+ 的定义及边界](../../docs/pa-plus-definitions.md)

类型论给了我们另一种陈述和复用数学的方法。简单高阶逻辑已经能把函数作为变量。Isabelle/HOL 的素数定理证明用到了狄利克雷级数和复分析。Lean 还能让类型依赖于值，例如矩阵的行数和列数可以写进类型。同一份证明也可以量化两个任意类型和它们之间的保运算映射。mathlib 已分别给出了费马大定理的数论陈述和用复 ζ 函数零点表述的黎曼假设。[Isabelle/HOL 的素数定理证明](https://isa-afp.org/entries/Prime_Number_Theorem.html) · [Lean 类型系统](https://lean-lang.org/doc/reference/latest/The-Type-System/) · [费马大定理的形式陈述](https://leanprover-community.github.io/mathlib4_docs/Mathlib/NumberTheory/FLT/Basic.html#FermatLastTheorem) · [黎曼假设的形式陈述](https://leanprover-community.github.io/mathlib4_docs/Mathlib/NumberTheory/LSeries/RiemannZeta.html#RiemannHypothesis)

## 两条可行路线

| 路线 | 起步时搜索能看见什么 | 何时值得选 |
| --- | --- | --- |
| Metamath 加上选定的 `set.mm` 主题 | 选定的公理、定义和引理。检查器加载它们的传递依赖，检索器只暴露实验允许的前提。 | 如果最终交付必须是可由 Metamath 回放的证书，这条路保留了验证工具和证书格式。高层策略仍需适配新领域。 |
| Lean 4 加上自建库 | 从很少的定义和推理规则起步，逐轮加入在训练集上自行证出的引理。mathlib 可以单独提供外部评测题。 | 适合研究一般代数结构、复函数、依赖于值的类型及带局部假设的中间命题。 |

Metamath 是检查规则应用的语言。用它写成的数据库包括集合论库 `set.mm` 和高阶逻辑库 `hol.mm`。仓库在一份 `set.mm` 快照上已经加载并验证了 50,888 个断言，50 次有限预算的生成尝试得到 3 个新声明。这次试验表明仓库可以读取大库并验证新证书。要判断学习是否有效，还需控制每个状态可调用的引理，以及采到的题目到底需要几种策略。[项目实测](../../docs/generic-metamath-generation.md) · [Metamath 集合论库](https://us.metamath.org/mpeuni/mmset.html) · [Metamath 高阶逻辑库](https://us.metamath.org/holuni/mmhol.html)

## 本机验证了什么

工具链固定为 Lean 4.36.0。下列数学示例只导入 `Init.Prelude`。[示例源码](Probe.lean) · [冻结题翻译器](check_frozen.py)

| 检查 | 结果 |
| --- | --- |
| 翻译 v2.1 的 36 道冻结测试题 | 逐条从原始记录解析公式，按自然数解释，36 条命题声明均通过类型检查。为 `p0004` 单独手写了 `∀ n, n = 1*n` 的证明，并按该行翻译出的实际目标核验。同一证明交给 `p0017` 时，Lean 报告类型不匹配。 |
| 拓展数学对象 | 另证明了后继与加法的恒等式、保单位元和乘法的映射保持自然数次幂、左分配律下的整除加法封闭性，以及任意元素类型的矩阵转置两次复原。还写出了费马大定理的自然数陈述。 |
| 复核证明和失败样例 | 以上五份证明都通过 Lean 内核，`#print axioms` 均报告无公理依赖。用 `leanexport` 导出这五份证明及依赖后，官方 `leanchecker` 和独立 Rust 检查器 `con-ron --verified` 均接受，后者检查了 46 个声明。误证 `0=1`、把 2×3 矩阵当作 3×2，以及开启警告即错后遗留的 `sorry`，均被拒绝。 |
| 从 Python 得到逐步反馈 | 另用 Lean 自带语言服务器运行算术题。Python 在同一进程先读到原目标，发出归纳并完成基例后，读到局部假设 `ih : k = 1*k` 和后继目标。补上最后一步后，返回的目标列表为空。 |

翻译器目前支持这批留出题实际用到的对象变量、量词、合取及算术运算，按作用域闭合自由变量。Metamath 的 `$d` 约束限制模式替换，并不表示两个自然数的值必须不同。这次翻译解释的是命题，没有把任何旧证明转换成 Lean 证明。Lean 原生的 `Nat` 证明也不会自动生成指定 PA 公理下的 Metamath 证书。若要在 Lean 内保留后一语义，可以把 PA 公式与推导规则定义为数据，并要求输出该公式的推导树。

在有历史冻结题文件的工作副本中，从仓库根目录运行以下命令。也可以单独进入本目录运行 `lean -DwarningAsError=true Probe.lean`。

```powershell
$env:PYTHONPATH = 'src;v2;v2_1'
python v3/lean/check_frozen.py outputs/abc-261122/frozen/cases.jsonl --witness-id p0004
```

要复验独立检查，可将 `Probe.lean` 和 `lean-toolchain` 复制到临时目录，在其中用 `lean -DwarningAsError=true -o Probe.olean Probe.lean` 生成模块，并把该目录加入 `LEAN_PATH`。用同一工具链的 `bin` 目录里的 `leanexport Probe -- V3Probe.natOneMul V3Probe.succAdd V3Probe.mapPower V3Probe.dividesAdd V3Probe.transposeTwice` 导出证明及其依赖，将标准输出按原始 UTF-8 字节保存为 `probe.ndjson`，再运行 `con-ron --verified probe.ndjson`。

## 怎样迁移到可比较的训练实验

1. **先接搜索。** 沿用 v2.1 的分题、选候选、记轨迹和按整题证明发奖励的设计。Lean 一侧需记录每个子目标的类型和局部假设，支持从同一状态尝试多个动作，再把选定的完整证明交给新的检查进程。旧的 Metamath 替换、作用域和 Vampire 证明重建都有专用实现。社区 REPL 和 Pantograph 提供机器接口，可用于下一步的交互式搜索。[当前搜索实现](../../v2_1/pa_prover_v21/search.py) · [Lean 社区 REPL](https://github.com/leanprover-community/repl) · [Pantograph 论文](https://arxiv.org/abs/2410.16429)
2. **把人类知识的用量记清。** 为初始定义、允许的基础规则和自证引理分别记录来源，固定每个训练轮次可见的引理。对完整证明检查其实际引用的常量和传递依赖，包括自动化暗中调用的引理。实测中，显式加入一条公理后，Lean 会接受依赖它的定理，`#print axioms` 能显示这条公理。两份证明也可能列出完全相同的公理，其中一份就直接调用了人类证明好的目标定理。判断数据来源需要这两层检查。HTPS 的 Lean 实验也使用了从 mathlib 抽取的监督样本。[Lean 证明验证](https://lean-lang.org/doc/reference/latest/ValidatingProofs/) · [独立内核与导出格式](https://lean-lang.org/doc/reference/latest/External-Kernels/) · [HTPS 的监督训练数据](https://proceedings.neurips.cc/paper_files/paper/2022/file/a8901c5e85fb8e1823bbf0f755053672-Paper-Conference.pdf)
3. **让不同的第一步都能派上用场。** 首轮可把 PA 共有算术、整除和同余、抽象代数中的运算与同态放进同一课程。上面的 `dividesAdd` 就演示了不同的解法。要从 `d ∣ a` 和 `d ∣ b` 推出 `d ∣ a+b`，从两条假设中取出因子，以它们的和作见证，再用分配律收尾。这类题给模型提供了归纳之外的决策。按证明结构留出测试族，让手写、随机和学习策略使用同一动作集、引理快照和时间预算，逐题统计双方各自独立解出几道，并抽查哪些分叉改变了结果。本次完成了表达、交互和验收试验，训练对照尚待实施。

## 从证明生成训练题

在 Lean 中，命题 `P` 的类型是 `Prop`。给出一个类型为 `P` 的项 `p` 就给出了证明。内核在声明的上下文和已接受的常量定义下检查 `p : P`。例如，函数应用要检查参数类型，再把实参代入结果类型。类型比较允许按定义计算，`(fun x => f x) a` 和 `f a` 在这里是同一个结果。归纳类型提供构造子和消去规则，`Nat.rec` 可以把基例和归纳步变成一个对所有自然数有效的证明。写 `by` 时，tactic 在 elaborator（负责补全隐含信息、构造证明项的程序）中运行，可以建立子目标、搜索引理并填入证明。最后的证明项交给内核检查。[Lean 类型与按定义相等](https://lean-lang.org/doc/reference/latest/The-Type-System/) · [表达式的细化与内核检查](https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/) · [Lean 的 tactic 教程](https://lean-lang.org/theorem_proving_in_lean4/Tactics/)

### 正向造题，反向解题

假设 `hxy : x = y` 和 `hyz : y = z`。把这两个证明交给 `Eq.trans`，得到 `x = z` 的证明。再用 `congrArg f`，得到 `f x = f z` 的证明。Lean 可以直接告诉我们这个项所证明的命题。

```lean
section
variable {R : Type} (f : R → R) (x y z : R) (hxy : x = y) (hyz : y = z)
#check congrArg f (Eq.trans hxy hyz)
-- 得到的类型是 f x = f z
end
```

正向生成时，记录每次调用哪条规则、使用哪些前提以及得到什么结论。解题时从根结论往下走，一次填一个证明空位。下表对应种子 17 的第 38 题，待证目标按 Lean 的处理顺序排列。

| 动作前的全部目标 | 给模型的动作标签 |
| --- | --- |
| `f x = f z` | `refine congrArg f (?_ : x = z)` |
| `x = z` | `refine Eq.trans (?_ : x = y) (?_ : y = z)` |
| `x = y`，`y = z` | `exact hxy` |
| `y = z` | `exact hyz` |

第二步包含一个真正的选择。只看 `x = z`，两端没有告诉模型应当用哪个中间量。参考证明选了 `y`，两个带类型的空位把这个选择记录进了动作。最后一步之后没有待证目标。

### 采样器具体做了什么

[生成器](generate.py) 用 Python 标准库采样，调用本目录固定版本的 Lean 验证。在一个共同的上下文中，有两个从 `R` 到 `R` 的函数、加法、乘法和左分配律假设。初始事实为 `x = y`、`y = z` 和 `d ∣ x`。这些假设都是定理的参数，发布的命题对每个满足假设的实例成立。

| 抽到的规则 | 能生成的新事实 |
| --- | --- |
| 传递性 | 选取中间项相同的 `a = b` 和 `b = c`，得到 `a = c`。 |
| 函数保相等 | 从 `a = b` 得到 `f a = f b`。也可选择 `g`、`mul d` 或 `add x`。 |
| 构造整除见证 | 随机生成一个类型为 `R` 的表达式 `t`，用因子 `t` 证明 `d ∣ d*t`。 |
| 两个倍数相加 | 从 `d ∣ a` 和 `d ∣ b` 得到 `d ∣ a+b`。这一步使用项目已有、由人写好并验证的 `V3Probe.dividesAdd`。 |
| 沿等式替换 | 从 `d ∣ a` 和 `a = b` 得到 `d ∣ b`，把等式和整除连接起来。 |

每轮随机选规则，再到事实池里取匹配的前提。当前小文法把项写成统一的括号形式，用结构相同的项匹配中间量。重复结论会跳过，新结论连同证明保留。深度小于 4 的结果可继续作前提。对于每个保留的根，程序生成完整证明项，再从根遍历证明树，生成可执行的 tactic 序列。Lean 分别检查两份证明，并在回放时逐步核对当前主目标。整个批次成功后才写入指定的结果目录。

从仓库根目录可用下面的命令生成同一批题。`--out` 指定一个新目录。

```powershell
python v3/lean/generate.py --seed 17 --count 48 --out outputs/lean-demo
```

`problems.jsonl` 含题号、公共上下文和目标。`solutions.jsonl` 通过题号关联证明项以及每一步的待证目标和动作。`Certified.lean` 可用固定版本的 Lean 单独回放。在本次实测中，48 个不同题面对应 48 份证明项、48 份 tactic 证明和 143 步动作。将这 48 份证明项及所需声明一起导出，独立 Rust 检查器 `con-ron` 也接受了全部 61 个声明。种子 17 的最后一条生成规则分布为函数保相等 16 题、整除见证 15 题、倍数相加 13 题、等式传递和沿等式替换各 2 题。

训练时可以从题面与当前全部目标预测下一步，将动作当作监督标签。评测只发题面，由策略自己提出中间项、见证或引理，并重跑整份证明计分。参考轨迹给出一条成功路线，其他能通过验证的路线也应得分。这份小语料共享起始事实和子证明，适合调试接口。比较学习策略与启发式时，应按证明结构或生成来源分组留出，并给双方同一引理库和预算。

已有的 Metamath 生成器会为一条断言寻找符号替换、类型证明和能满足前提的父节点。正向组合的思想可以保留。对于一般的 Lean 引理，例如 `r : ∀ t, P t → Q t`，先把参数设为待定的 `?x`，在池中找到 `p : P a`，让 Lean 匹配 `P ?x` 和 `P a`。若得出 `?x := a`，就能把证明项 `r a p` 及推断出的结论 `Q a` 加入池。实际引理还可能有隐式类型参数、类型类实例和多个前提，可以交由 Lean 的元编程接口处理。[Metamath 的生成算法](../../docs/generic-metamath-generation.md) · [Lean 元编程中的类型推断与统一](https://leanprover-community.github.io/lean4-metaprogramming-book/main/04_metam.html)

扩展到存在命题时要生成见证及其性质证明。扩展到归纳时要在增加了归纳变量和归纳假设的局部上下文中生成归纳步。另一条数据来源是先独立采样猜想，再让求解器限时寻找证明，把成功样本注入训练。预算耗尽的猜想保留为未知。
