# PA+ 分层定义库与生成接口

本文说明 [pa-plus-definitions.json](../formal/pa-plus-definitions.json) 和由它机械生成的
[peano-pa-plus.mm](../formal/peano-pa-plus.mm)。这一扩展继续采用 PA+/Metamath 路线，
没有增加新的对象域、未证明公理或真实数类型。

当前第一批目录包含：

- 68 个保守高层谓词定义；
- 35 个闭合的常见定理目标公式；
- 数论、有限递推、算术函数、整数、有理数和初等分析证书六类接口；
- 自动生成的量词新鲜性 `$d` 条件；
- 基础库不变性、自由变量、循环依赖、前向依赖、精确语法声明和目标类型审计。

这里的“目标公式”只有 `statement` 类型。它们用于证明能力评测和确认表达覆盖，
不会进入逻辑断言、规则或已证明定理数据库。

## 1. 文件职责

| 文件 | 职责 |
| --- | --- |
| `formal/pa-plus-definitions.json` | 供人和 AI 编辑的紧凑定义目录 |
| `src/metamath_generator/definitions.py` | 编译、保守性审计和命令行入口 |
| `formal/peano-pa-plus.mm` | 生成后可由项目解析器和 Metamath 工具处理的形式源 |
| `tests/test_pa_plus.py` | 确定性生成、闭合性、分层和可信边界回归测试 |

重新生成：

```powershell
$env:PYTHONPATH='src'
python -m metamath_generator.definitions `
  formal/pa-plus-definitions.json formal/peano-pa-plus.mm
```

成功命令会输出定义数、目标公式数以及最大定义 AST 大小。编译器先生成临时文件，
只有完整解析和审计通过才替换正式 `.mm` 文件。

## 2. 保守性契约

每个新谓词 `P` 只能以如下形式出现：

```text
df-P $a |- iff P(parameters...) OLD_OR_EARLIER_FORMULA $.
```

审计器强制检查：

1. 所有新逻辑断言都以 `df-` 开头；
2. 定义根节点必须是 `iff`；
3. 左侧是一次完整的谓词应用，参数互不重复；
4. 右侧不得出现正在定义的谓词；
5. 右侧只能依赖基础语言或目录中更早的定义；
6. 右侧自由变量必须来自左侧参数；
7. 所有量词变量与参数、其他量词变量具有 `$d` 新鲜性约束；
8. 每个谓词恰好有一个对应的 `wff_` 语法声明；
9. 目录声明的定义和实际生成的定义集合完全一致；
10. 新常量、新语法规则和非逻辑声明都与目录逐项完全一致，不能夹带额外声明；
11. 基础库的语句、假设、类型、`$d` 约束和原始 token 必须逐项保持不变；
12. 左侧参数必须都是 `term`，不得带有会暗中限制实例化的参数间 `$d` 约束；
13. 参数必须在右侧实际出现；唯一例外是目录显式列入
    `allow_unused_parameters` 的表示参数；
14. 目标公式闭合，只能是非逻辑的 `statement`，而且不得注册为可递归使用的
    语法规则。

这证明的是“新谓词可按无环定义消去”的语法保守性。审计不能代替对定义右侧是否
准确表达预期数学概念的人工语义审阅，因此目录同时保存 `summary`、`area` 和
`theorem_families`。

## 3. 定义分层

### 3.1 自然数序和基础关系

```text
natne positive ge gt between natdiff minrel maxrel
```

这些定义消除常见公式中反复出现的否定等式、反向序、闭区间和受限减法包装。

### 3.2 初等数论

```text
properdivides composite coprime gcdrel lcmrel congruent remainder
square cube sumtwosquares sumfoursquares pythagorean pellsolution
quadraticresidue squarefree valuation perfectpower multiplicativeorder
mersenne fermatnumber
```

`gcdrel` 和 `lcmrel` 使用通用整除性质，而不是未经证明的新函数符号。
`congruent` 用两侧增加模数倍数的等式避免引入整数减法，并显式要求模数为正。
`multiplicativeorder A M K` 只在 `M > 1` 且 `A` 与 `M` 互素时成立，排除了模
0、模 1 和非单位的非标准“阶”。

### 3.3 有限序列与递推

```text
seqat seqsum seqprod factorial fibonacci binomial
arithterm geomterm arithsum geomseries
```

底层仍使用 Gödel β 关系，因此没有削弱严谨性。但只有 `df-seqat` 直接提到
`beta`；其余定义和模型侧状态只使用 `seqat`、`seqsum`、`factorial` 等高层接口。
这样把编码细节集中在单一可审计边界，同时减少训练数据中的重复展开。

`binomial N K R` 采用标准的全定义约定：`K <= N` 时由阶乘关系刻画，`K > N`
时 `R = 0`。这样 Pascal 公式等陈述不再依赖一个未写出的定义域前提。

### 3.4 算术函数

```text
divisorcount divisorsum eulerphi
perfectnumber abundantnumber deficientnumber
```

计数与累加通过隐藏的认证有限递推轨迹定义。它们是关系而不是默认具有唯一性的
函数；需要使用函数性时，必须另行证明存在唯一性。`divisorcount`、`divisorsum`
和 `eulerphi` 的定义域显式限制为正整数，暂不对输入 0 赋值。这避免把
“0 的正因子个数”等非标准约定带进目标公式。

### 3.5 整数和有符号有理数

整数 `a-b` 用自然数对 `(a,b)` 表示：

```text
inteq intle intlt intadd intmul intneg intabs
```

有理数 `(a-b)/c` 用两个自然数和正分母表示：

```text
qvalid qeq qle qlt qadd qmul qneg qabs
```

等式和运算都通过交叉相乘及整数对关系定义，不要求分数约分。不同表示的等价性
由 `inteq`/`qeq` 明确表达。`qvalid A B C` 的 `A`、`B` 只表示任意有符号分子，
所以该定义有意只检查正分母 `C`；这两个未使用参数是目录中唯一显式白名单例外。

### 3.6 初等分析证书

```text
ratbetween ratabsle ratabslt
sqrtlower sqrtupper sqrtinterval
nthrootlower nthrootupper nthrootexact
```

这些谓词只表达可由自然数算术检查的有理区间和误差证书。例如：

```text
sqrtlower X A B  :=  B > 0 and A^2 < X B^2
sqrtupper X A B  :=  B > 0 and X B^2 < A^2
```

它们不会把 `sqrt(X)` 声明为一个新的实数对象。已有 `lnlower`、`lnupper`、
`lilower` 和 `liupper` 同样采用有理切证书。

## 4. 当前目标公式覆盖

第一批 35 个闭合公式覆盖以下常见结论族：

- 素数无穷、除法算法、素因子存在；
- gcd/lcm 存在及乘积关系、Bézout、二模数 CRT；
- Fermat 小定理、Euler 定理、Wilson 定理；
- 二平方、四平方、Pell 方程、平方自由分解；
- 乘法阶整除 φ、除数个数奇偶、素数为亏数、偶完全数构造；
- 阶乘和 Fibonacci 递推、二项式对称与 Pascal 公式；
- 等差和、等比和；
- 整数和有理数三角不等式、有理数稠密性；
- Archimedean 界、平方根有理区间、素数平方根无理性；
- n 次根的有理判据、根区间次序、绝对误差单调性。

目录中的 `requires` 必须与目标公式 AST 实际使用的新定义完全一致。这样既能生成
领域覆盖清单，也能防止元数据声称依赖某个接口而公式实际绕过它。

## 5. 对 AI 生成与训练的接口

模型可把新谓词视为原子形式符号，避免每次书写全部展开。当前实现用以下桥作为
搜索宏（下例为逻辑结构示意，不是 CLI 或动作 token 格式）：

```text
df-gcdrel             : |- iff (gcdrel A B G) expansion
gen_df_gcdrel_unfold  : |- implies (gcdrel A B G) expansion
gen_df_gcdrel_fold    : |- implies expansion (gcdrel A B G)
```

`df-gcdrel` 本身证明等价式，不是直接证明任意 `gcdrel A B G` 的 tactic。桥由
`bi1`/`bi2` 和 `ax-mp` 推出，命名已稳定为 `gen_df_<谓词>_<方向>`；可用于有界
实例化和后续组合，证书中会内联。当前基础语料已有生成类型、引导目标和定义
依赖元数据。完整训练/搜索数据设计仍应保留：

- 高层目标 AST；
- 使用的定义 ID；
- 定义依赖闭包；
- 必要时的展开后 AST；
- 最终 Metamath 证书。

建议优先在高层接口上训练策略，在证书编译或受控的定义展开动作中才进入 β 编码
和交叉相乘细节。tokenizer 已按每个 Metamath 符号使用原子 token，因此新增谓词
天然形成稳定词表项。

已有 checkpoint 必须使用追加式 `upgrade-pa-plus`，不能仅因符号相同就混用重新
构建的 token ID。当前实现、已验证路径和未接通接口见
[PA+ 神经训练与闭环推理](pa-plus-neural-training.md)。

## 6. 严格边界

### 6.1 定义不是已证明函数

`factorial N R`、`gcdrel A B G` 等首先只是关系。右侧设计为函数图并不自动提供：

```text
forall N exists unique R, factorial N R
```

存在性、唯一性、递推方程和代数性质仍需要正式证明。未来只有在这些定理完成后，
才能考虑增加可消去的函数记号层。

### 6.2 初等分析不是完整实分析

PA 的对象仍只有自然数。任意实数集合不可由单个自然数编码，因此本路线当前只
覆盖：

- 有理数和有理误差；
- 有限递推与有限和积；
- 特定可计算函数的上下界证书；
- 用显式 ε/有理切改写的结论。

涉及任意实函数、完备性、不可数对象、一般测度和泛函空间的定理不应伪装成当前
PA+ 已覆盖。若未来必须覆盖这些领域，应建立更强的保守编码层或单独的 Lean 后端。

### 6.3 `statement` 不是真理声明

目录中的目标公式用于表达覆盖和基准构造，尚未附带 `$p` 证明。解析成功只说明：

- 公式语法正确；
- 变量全部闭合；
- 所需定义已经存在；
- 公式没有作为逻辑公理进入系统。

它不说明该公式已经在 PA 中证明。

解析器只把形如 `statement <wff变量>` 的基础包装规则注册为语法规则；具体目标
声明不会再把自身注册为语法产生式。因此类似 `bad $a statement 0 $.` 的畸形输入
不能依靠“先注册自己、再验证自己”通过检查。

### 6.4 独立验证的准确含义

生成文件除项目解析器审计外，也用官方发行包中的 C Metamath `0.199.pre` 独立读取
过（可执行文件 SHA-256：`A9FA7F12EFA5535609D95B533E28A4A83C90EB7AB43ADF1DE5D7AAE0DE87C0BF`）：
完整包含链共 73,229 字节、1,412 条语句，验证器报告没有源文件错误。当前扩展包含
282 条 `$a` 且没有 `$p`，因此 `VERIFY PROOF *` 成功只确认现有证明集合可遍历，
不能把 35 个目标公式误报为已经证明。定义的数学语义以及后续 `$p` 证明仍需分别
审查。

## 7. 后续扩展原则

新增定义时应按以下顺序：

1. 先在目录中说明数学语义和目标定理族；
2. 尽量复用已有高层谓词，避免再次直接使用 `beta`；
3. 对多值关系说明定义域和表示不唯一性；
4. 加入至少一个闭合 `statement` 证明新接口确实能表达目标；
5. 重新生成 `.mm` 并运行保守性测试；
6. 对定义右侧进行独立人工语义审阅；
7. 再把新符号加入训练语料和 benchmark。

下一批候选包括一般有限区间计数、组合数的递推式定义、素数指数分解证书、Mobius
函数的有符号图、连分数、更多对数/指数有理逼近，以及按证明骨架隔离的目标集。
