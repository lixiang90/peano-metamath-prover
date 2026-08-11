# `peano.mm` 数论与有理估计扩展

[peano-number-theory.mm](../formal/peano-number-theory.mm) 是
[peano.mm](../formal/peano.mm) 的保守词汇扩展。它只增加语法和显式定义，不把费马
大定理、哥德巴赫猜想、素数定理或黎曼猜想直接假设为真。

## 三类声明

扩展文件严格区分：

1. `wff_*`：新谓词的语法构造规则；
2. `df-*`：形如 `|- iff NEW(...) OLD_FORMULA` 的显式定义；
3. `*-statement`：类型为 `statement` 的命名公式，不是 `|-` 断言。

因此下列标签不会进入定理数据库：

```text
flt-statement
goldbach-statement
pnt-statement
riemann-von-koch-statement
```

## 自然数关系

| 谓词 | 含义 |
|---|---|
| `le A B` | `A ≤ B` |
| `divides A B` | `A` 整除 `B` |
| `prime A` | `A` 是素数 |
| `even A` | `A` 是偶数 |
| `beta B C I V` | Gödel β 编码序列的第 `I` 项为 `V` |
| `pow A N R` | `R = A^N` |
| `primecount X N` | `N = π(X)` |

`pow` 和 `primecount` 都不是未经约束的新函数符号。它们是通过 β 编码有限
递归轨迹定义的三元/二元关系。

## 有理数与分析量

有理数 `A/B` 用两个自然数表示，并要求 `B > 0`。`ratle`、`ratlt`、
`ratadd` 和 `ratsub` 全部使用交叉相乘定义。

自然对数采用恒等式

```text
log X = 2 Σ ((X-1)/(X+1))^(2k+1)/(2k+1)
```

的有限部分和及几何尾项上界：

- `lnlower X A B`：`A/B < log X`；
- `lnupper X A B`：`log X < A/B`。

`Li(X)=∫₂ˣdt/log(t)` 不作为实数对象加入。`lilower` 和 `liupper` 使用
有理均匀分割、单调被积函数的左右矩形以及 β 编码的有理累加器，定义其
严格有理上下切。

## 四个命名公式

### 费马大定理

对 `N>2` 及正自然数 `A,B,C`：

```text
A^N + B^N ≠ C^N
```

幂通过 `pow` 关系表达。

### 强哥德巴赫猜想

每个大于 2 的偶数都是两个素数之和。

### 素数定理

不使用极限或 `~`，而写成：

```text
对每个正有理数 E/D，
存在 N≥2，使所有 X≥N 满足
|π(X) log(X)/X - 1| < E/D。
```

`pntat` 将绝对值拆成两个严格有理不等式，并用 `lnlower/lnupper` 给出
可核验的对数界。

### 黎曼猜想的 von Koch 形式

大 O 被完全展开为：

```text
存在自然数 C>0 和 N≥2，使所有 X≥N 满足
|π(X)-Li(X)| ≤ C sqrt(X) log(X)。
```

`rhat` 使用 `lilower/liupper` 和 `lnlower`；平方根通过两边平方消去，
最终只剩自然数乘法与有理数交叉不等式。

## 使用

```powershell
python -m metamath_generator formal/peano-number-theory.mm --mode random --steps 1000
```

检查扩展：

```powershell
python -m unittest discover -s tests -v
```

这里完成的是“可表达性和保守定义层”，不是四个著名结果的形式证明。
`log`、`Li` 的证书关系与通常分析定义等价，仍需要在更丰富的形式化分析
库中证明；扩展本身没有把这种外部正确性声明成新的 `|-` 公理。
