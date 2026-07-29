$(
  Conservative number-theory and rational-estimate vocabulary for peano.mm.

  This file deliberately separates three kinds of declarations:

  1. wff_* declarations add syntax only.
  2. df-* declarations are explicit definitions of fresh predicates in terms
     of the old language or predicates defined earlier in this file.
  3. flt-statement, goldbach-statement, pnt-statement, and
     riemann-von-koch-statement have type "statement", not "|-".  They name
     formulas without asserting them as axioms or proved theorems.

  Natural numbers are the only objects.  Rationals are represented by pairs
  (numerator, positive denominator).  Finite sequences are represented with
  Goedel's beta relation.  log and Li are represented by rational lower and
  upper certificates; no real-number type is added.
$)

$[ peano.mm $]

$(
##############################################################################
  Additional metavariables
##############################################################################
$)

$v a b c d e f g h i j k l m n p q r $.
ta $f term a $.
tb $f term b $.
tc $f term c $.
td $f term d $.
te $f term e $.
tf $f term f $.
tg $f term g $.
th $f term h $.
ti $f term i $.
tj $f term j $.
tk $f term k $.
tl $f term l $.
tm $f term m $.
tn $f term n $.
tp $f term p $.
tq $f term q $.
tr $f term r $.

$v x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 $.
varx0 $f var x0 $.
varx1 $f var x1 $.
varx2 $f var x2 $.
varx3 $f var x3 $.
varx4 $f var x4 $.
varx5 $f var x5 $.
varx6 $f var x6 $.
varx7 $f var x7 $.
varx8 $f var x8 $.
varx9 $f var x9 $.
varx10 $f var x10 $.
varx11 $f var x11 $.
varx12 $f var x12 $.
varx13 $f var x13 $.
varx14 $f var x14 $.
varx15 $f var x15 $.

$v theta $.
wff_theta $f wff theta $.

$(
##############################################################################
  Syntax
##############################################################################
$)

$c statement $.
statement_of $a statement theta $.

$c le divides prime even beta pow primecount $.
wff_le $a wff le a b $.
wff_divides $a wff divides a b $.
wff_prime $a wff prime a $.
wff_even $a wff even a $.
wff_beta $a wff beta a b c d $.
wff_pow $a wff pow a b c $.
wff_primecount $a wff primecount a b $.

$c ratle ratlt ratadd ratsub $.
wff_ratle $a wff ratle a b c d $.
wff_ratlt $a wff ratlt a b c d $.
wff_ratadd $a wff ratadd a b c d e f $.
wff_ratsub $a wff ratsub a b c d e f $.

$c logterm logsumstep logsum logtail lnlower lnupper lnqlower lnqupper $.
wff_logterm $a wff logterm a b c d $.
wff_logsumstep $a wff logsumstep a b c d e f $.
wff_logsum $a wff logsum a b c d $.
wff_logtail $a wff logtail a b c d $.
wff_lnlower $a wff lnlower a b c $.
wff_lnupper $a wff lnupper a b c $.
wff_lnqlower $a wff lnqlower a b c d $.
wff_lnqupper $a wff lnqupper a b c d $.

$c licertlowerstep licertupperstep licertlower licertupper lilower liupper $.
wff_licertlowerstep $a wff licertlowerstep a b c d e f $.
wff_licertupperstep $a wff licertupperstep a b c d e f $.
wff_licertlower $a wff licertlower a b c d $.
wff_licertupper $a wff licertupper a b c d $.
wff_lilower $a wff lilower a b c $.
wff_liupper $a wff liupper a b c $.

$c pntat natminusrootle ratminusnatrootle rhat $.
wff_pntat $a wff pntat a b c $.
wff_natminusrootle $a wff natminusrootle a b c d e f g $.
wff_ratminusnatrootle $a wff ratminusnatrootle a b c d e f g $.
wff_rhat $a wff rhat a b $.

$(
##############################################################################
  Elementary explicit definitions
##############################################################################
$)

df-le $a |- iff le a b or < a b = a b $.

${
  $d x0 a $.
  $d x0 b $.
  df-divides $a |- iff divides a b
    exists x0 = b * a x0 $.
$}

${
  $d x0 a $.
  df-prime $a |- iff prime a
    and < S 0 a
      forall x0 implies divides x0 a
        or = x0 S 0 = x0 a $.
$}

${
  $d x0 a $.
  df-even $a |- iff even a
    exists x0 = a + x0 x0 $.
$}

$(
  beta B C I V means that V is the remainder of B modulo
  1 + (I + 1) C.  It is an old-language formula and is the sequence-coding
  primitive used by every recursive relation below.
$)
${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x0 d $.
  df-beta $a |- iff beta a b c d
    and < d S * S c b
      exists x0 = a + * x0 S * S c b d $.
$}

$(
  pow A N R is the graph of natural exponentiation.  The beta-coded sequence
  starts at 1 and each next entry is the previous entry multiplied by A.
$)
${
  $d x0 a $. $d x0 b $. $d x0 c $.
  $d x1 a $. $d x1 b $. $d x1 c $.
  $d x2 a $. $d x2 b $. $d x2 c $.
  $d x3 a $. $d x3 b $. $d x3 c $.
  $d x4 a $. $d x4 b $. $d x4 c $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $. $d x0 x4 $.
  $d x1 x2 $. $d x1 x3 $. $d x1 x4 $.
  $d x2 x3 $. $d x2 x4 $. $d x3 x4 $.
  df-pow $a |- iff pow a b c
    exists x3 exists x4
      and beta x3 x4 0 S 0
        and beta x3 x4 b c
          forall x0 implies < x0 b
            exists x1 exists x2
              and beta x3 x4 x0 x1
                and beta x3 x4 S x0 x2
                  = x2 * x1 a $.
$}

$(
  primecount X N says N is pi(X), counting primes not exceeding X.  The
  beta-coded counter starts at zero and increments exactly when S(I) is prime.
$)
${
  $d x0 a $. $d x0 b $.
  $d x1 a $. $d x1 b $.
  $d x2 a $. $d x2 b $.
  $d x3 a $. $d x3 b $.
  $d x4 a $. $d x4 b $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $. $d x0 x4 $.
  $d x1 x2 $. $d x1 x3 $. $d x1 x4 $.
  $d x2 x3 $. $d x2 x4 $. $d x3 x4 $.
  df-primecount $a |- iff primecount a b
    exists x3 exists x4
      and beta x3 x4 0 0
        and beta x3 x4 a b
          forall x0 implies < x0 a
            exists x1 exists x2
              and beta x3 x4 x0 x1
                and beta x3 x4 S x0 x2
                  and implies prime S x0 = x2 S x1
                    implies not prime S x0 = x2 x1 $.
$}

$(
##############################################################################
  Nonnegative rational arithmetic

  A rational is represented by A/B with B > 0.  Fractions need not be reduced.
##############################################################################
$)

df-ratle $a |- iff ratle a b c d
  and < 0 b
    and < 0 d
      le * a d * c b $.

df-ratlt $a |- iff ratlt a b c d
  and < 0 b
    and < 0 d
      < * a d * c b $.

df-ratadd $a |- iff ratadd a b c d e f
  and < 0 b
    and < 0 d
      and < 0 f
        = * e * b d
          * f + * a d * c b $.

df-ratsub $a |- iff ratsub a b c d e f
  ratadd c d e f a b $.

$(
##############################################################################
  Rational certificates for the natural logarithm

  For X = M + 1, put z = M / (X + 1).  The identity

    log X = 2 sum_{k >= 0} z^(2k+1) / (2k+1)

  has positive terms.  logsum is a rational partial sum and logtail is the
  elementary geometric upper bound

    2 z^(2K+1) / ((2K+1)(1-z^2)).

  Thus lnlower and lnupper are arithmetically checkable rational cuts.
##############################################################################
$)

${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $.
  $d x1 a $. $d x1 b $. $d x1 c $. $d x1 d $.
  $d x2 a $. $d x2 b $. $d x2 c $. $d x2 d $.
  $d x0 x1 $. $d x0 x2 $. $d x1 x2 $.
  df-logterm $a |- iff logterm a b c d
    exists x0 exists x1 exists x2
      and = a S x0
        and pow x0 S + b b x1
          and pow S a S + b b x2
            and < 0 d
              = * c * x2 S + b b
                * d * S S 0 x1 $.
$}

$(
  One rational-accumulator step.  B/C and D/E are beta codes for
  numerator and denominator sequences; F is the current index.
$)
${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $. $d x0 e $. $d x0 f $.
  $d x1 a $. $d x1 b $. $d x1 c $. $d x1 d $. $d x1 e $. $d x1 f $.
  $d x2 a $. $d x2 b $. $d x2 c $. $d x2 d $. $d x2 e $. $d x2 f $.
  $d x3 a $. $d x3 b $. $d x3 c $. $d x3 d $. $d x3 e $. $d x3 f $.
  $d x4 a $. $d x4 b $. $d x4 c $. $d x4 d $. $d x4 e $. $d x4 f $.
  $d x5 a $. $d x5 b $. $d x5 c $. $d x5 d $. $d x5 e $. $d x5 f $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $. $d x0 x4 $. $d x0 x5 $.
  $d x1 x2 $. $d x1 x3 $. $d x1 x4 $. $d x1 x5 $.
  $d x2 x3 $. $d x2 x4 $. $d x2 x5 $.
  $d x3 x4 $. $d x3 x5 $. $d x4 x5 $.
  df-logsumstep $a |- iff logsumstep a b c d e f
    exists x0 exists x1 exists x2 exists x3 exists x4 exists x5
      and beta b c f x0
        and beta d e f x1
          and beta b c S f x2
            and beta d e S f x3
              and logterm a f x4 x5
                ratadd x0 x1 x4 x5 x2 x3 $.
$}

${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $.
  $d x1 a $. $d x1 b $. $d x1 c $. $d x1 d $.
  $d x2 a $. $d x2 b $. $d x2 c $. $d x2 d $.
  $d x3 a $. $d x3 b $. $d x3 c $. $d x3 d $.
  $d x4 a $. $d x4 b $. $d x4 c $. $d x4 d $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $. $d x0 x4 $.
  $d x1 x2 $. $d x1 x3 $. $d x1 x4 $.
  $d x2 x3 $. $d x2 x4 $. $d x3 x4 $.
  df-logsum $a |- iff logsum a b c d
    exists x1 exists x2 exists x3 exists x4
      and beta x1 x2 0 0
        and beta x3 x4 0 S 0
          and beta x1 x2 b c
            and beta x3 x4 b d
              forall x0 implies < x0 b
                logsumstep a x1 x2 x3 x4 x0 $.
$}

${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $.
  $d x1 a $. $d x1 b $. $d x1 c $. $d x1 d $.
  $d x2 a $. $d x2 b $. $d x2 c $. $d x2 d $.
  $d x3 a $. $d x3 b $. $d x3 c $. $d x3 d $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $.
  $d x1 x2 $. $d x1 x3 $. $d x2 x3 $.
  df-logtail $a |- iff logtail a b c d
    exists x0 exists x1 exists x2 exists x3
      and = a S x0
        and pow x0 S + b b x1
          and pow S a S + b b x2
            and = * S a S a + * x0 x0 x3
              and < 0 d
                = * c * * x2 S + b b x3
                  * d * * S S 0 x1 * S a S a $.
$}

${
  $d x0 a $. $d x0 b $. $d x0 c $.
  $d x1 a $. $d x1 b $. $d x1 c $.
  $d x2 a $. $d x2 b $. $d x2 c $.
  $d x0 x1 $. $d x0 x2 $. $d x1 x2 $.
  df-lnlower $a |- iff lnlower a b c
    exists x0 exists x1 exists x2
      and logsum a x0 x1 x2
        ratlt b c x1 x2 $.
$}

${
  $d x0 a $. $d x0 b $. $d x0 c $.
  $d x1 a $. $d x1 b $. $d x1 c $.
  $d x2 a $. $d x2 b $. $d x2 c $.
  $d x3 a $. $d x3 b $. $d x3 c $.
  $d x4 a $. $d x4 b $. $d x4 c $.
  $d x5 a $. $d x5 b $. $d x5 c $.
  $d x6 a $. $d x6 b $. $d x6 c $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $. $d x0 x4 $. $d x0 x5 $. $d x0 x6 $.
  $d x1 x2 $. $d x1 x3 $. $d x1 x4 $. $d x1 x5 $. $d x1 x6 $.
  $d x2 x3 $. $d x2 x4 $. $d x2 x5 $. $d x2 x6 $.
  $d x3 x4 $. $d x3 x5 $. $d x3 x6 $.
  $d x4 x5 $. $d x4 x6 $. $d x5 x6 $.
  df-lnupper $a |- iff lnupper a b c
    exists x0 exists x1 exists x2 exists x3 exists x4 exists x5 exists x6
      and logsum a x0 x1 x2
        and logtail a x0 x3 x4
          and ratadd x1 x2 x3 x4 x5 x6
            ratlt x5 x6 b c $.
$}

$(
  Bounds for log(A/B) = log A - log B.  These are only used at rational
  partition points A/B >= 2, so the nonnegative rational subtraction relation
  is sufficient.
$)
${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $.
  $d x1 a $. $d x1 b $. $d x1 c $. $d x1 d $.
  $d x2 a $. $d x2 b $. $d x2 c $. $d x2 d $.
  $d x3 a $. $d x3 b $. $d x3 c $. $d x3 d $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $.
  $d x1 x2 $. $d x1 x3 $. $d x2 x3 $.
  df-lnqlower $a |- iff lnqlower a b c d
    exists x0 exists x1 exists x2 exists x3
      and lnlower a x0 x1
        and lnupper b x2 x3
          ratsub x0 x1 x2 x3 c d $.
$}

${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $.
  $d x1 a $. $d x1 b $. $d x1 c $. $d x1 d $.
  $d x2 a $. $d x2 b $. $d x2 c $. $d x2 d $.
  $d x3 a $. $d x3 b $. $d x3 c $. $d x3 d $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $.
  $d x1 x2 $. $d x1 x3 $. $d x2 x3 $.
  df-lnqupper $a |- iff lnqupper a b c d
    exists x0 exists x1 exists x2 exists x3
      and lnupper a x0 x1
        and lnlower b x2 x3
          ratsub x0 x1 x2 x3 c d $.
$}

$(
##############################################################################
  Rational lower and upper certificates for Li(X) = integral_2^X dt/log(t)

  A positive mesh M gives the uniform rational partition

    2, 2 + 1/M, ..., X.

  Since 1/log(t) decreases for t >= 2, right endpoints give lower rectangles
  and left endpoints give upper rectangles.  The step predicates accumulate
  these rational rectangles in beta-coded numerator/denominator sequences.
  Existence of increasingly fine certificates defines the strict rational
  lower and upper cuts of Li(X), entirely inside natural-number arithmetic.
##############################################################################
$)

${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $. $d x0 e $. $d x0 f $.
  $d x1 a $. $d x1 b $. $d x1 c $. $d x1 d $. $d x1 e $. $d x1 f $.
  $d x2 a $. $d x2 b $. $d x2 c $. $d x2 d $. $d x2 e $. $d x2 f $.
  $d x3 a $. $d x3 b $. $d x3 c $. $d x3 d $. $d x3 e $. $d x3 f $.
  $d x4 a $. $d x4 b $. $d x4 c $. $d x4 d $. $d x4 e $. $d x4 f $.
  $d x5 a $. $d x5 b $. $d x5 c $. $d x5 d $. $d x5 e $. $d x5 f $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $. $d x0 x4 $. $d x0 x5 $.
  $d x1 x2 $. $d x1 x3 $. $d x1 x4 $. $d x1 x5 $.
  $d x2 x3 $. $d x2 x4 $. $d x2 x5 $.
  $d x3 x4 $. $d x3 x5 $. $d x4 x5 $.
  df-licertlowerstep $a |- iff licertlowerstep a b c d e f
    exists x0 exists x1 exists x2 exists x3 exists x4 exists x5
      and beta b c f x0
        and beta d e f x1
          and beta b c S f x2
            and beta d e S f x3
              and lnqupper + * S S 0 a S f a x4 x5
                ratadd x0 x1 x5 * a x4 x2 x3 $.
$}

${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $. $d x0 e $. $d x0 f $.
  $d x1 a $. $d x1 b $. $d x1 c $. $d x1 d $. $d x1 e $. $d x1 f $.
  $d x2 a $. $d x2 b $. $d x2 c $. $d x2 d $. $d x2 e $. $d x2 f $.
  $d x3 a $. $d x3 b $. $d x3 c $. $d x3 d $. $d x3 e $. $d x3 f $.
  $d x4 a $. $d x4 b $. $d x4 c $. $d x4 d $. $d x4 e $. $d x4 f $.
  $d x5 a $. $d x5 b $. $d x5 c $. $d x5 d $. $d x5 e $. $d x5 f $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $. $d x0 x4 $. $d x0 x5 $.
  $d x1 x2 $. $d x1 x3 $. $d x1 x4 $. $d x1 x5 $.
  $d x2 x3 $. $d x2 x4 $. $d x2 x5 $.
  $d x3 x4 $. $d x3 x5 $. $d x4 x5 $.
  df-licertupperstep $a |- iff licertupperstep a b c d e f
    exists x0 exists x1 exists x2 exists x3 exists x4 exists x5
      and beta b c f x0
        and beta d e f x1
          and beta b c S f x2
            and beta d e S f x3
              and lnqlower + * S S 0 a f a x4 x5
                ratadd x0 x1 x5 * a x4 x2 x3 $.
$}

${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $.
  $d x1 a $. $d x1 b $. $d x1 c $. $d x1 d $.
  $d x2 a $. $d x2 b $. $d x2 c $. $d x2 d $.
  $d x3 a $. $d x3 b $. $d x3 c $. $d x3 d $.
  $d x4 a $. $d x4 b $. $d x4 c $. $d x4 d $.
  $d x5 a $. $d x5 b $. $d x5 c $. $d x5 d $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $. $d x0 x4 $. $d x0 x5 $.
  $d x1 x2 $. $d x1 x3 $. $d x1 x4 $. $d x1 x5 $.
  $d x2 x3 $. $d x2 x4 $. $d x2 x5 $.
  $d x3 x4 $. $d x3 x5 $. $d x4 x5 $.
  df-licertlower $a |- iff licertlower a b c d
    exists x0 exists x1 exists x2 exists x3 exists x4
      and < 0 b
        and = a + S S 0 x0
          and beta x1 x2 0 0
            and beta x3 x4 0 S 0
              and beta x1 x2 * x0 b c
                and beta x3 x4 * x0 b d
                  forall x5 implies < x5 * x0 b
                    licertlowerstep b x1 x2 x3 x4 x5 $.
$}

${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $.
  $d x1 a $. $d x1 b $. $d x1 c $. $d x1 d $.
  $d x2 a $. $d x2 b $. $d x2 c $. $d x2 d $.
  $d x3 a $. $d x3 b $. $d x3 c $. $d x3 d $.
  $d x4 a $. $d x4 b $. $d x4 c $. $d x4 d $.
  $d x5 a $. $d x5 b $. $d x5 c $. $d x5 d $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $. $d x0 x4 $. $d x0 x5 $.
  $d x1 x2 $. $d x1 x3 $. $d x1 x4 $. $d x1 x5 $.
  $d x2 x3 $. $d x2 x4 $. $d x2 x5 $.
  $d x3 x4 $. $d x3 x5 $. $d x4 x5 $.
  df-licertupper $a |- iff licertupper a b c d
    exists x0 exists x1 exists x2 exists x3 exists x4
      and < 0 b
        and = a + S S 0 x0
          and beta x1 x2 0 0
            and beta x3 x4 0 S 0
              and beta x1 x2 * x0 b c
                and beta x3 x4 * x0 b d
                  forall x5 implies < x5 * x0 b
                    licertupperstep b x1 x2 x3 x4 x5 $.
$}

${
  $d x0 a $. $d x0 b $. $d x0 c $.
  $d x1 a $. $d x1 b $. $d x1 c $.
  $d x2 a $. $d x2 b $. $d x2 c $.
  $d x0 x1 $. $d x0 x2 $. $d x1 x2 $.
  df-lilower $a |- iff lilower a b c
    exists x0 exists x1 exists x2
      and licertlower a x0 x1 x2
        ratlt b c x1 x2 $.
$}

${
  $d x0 a $. $d x0 b $. $d x0 c $.
  $d x1 a $. $d x1 b $. $d x1 c $.
  $d x2 a $. $d x2 b $. $d x2 c $.
  $d x0 x1 $. $d x0 x2 $. $d x1 x2 $.
  df-liupper $a |- iff liupper a b c
    exists x0 exists x1 exists x2
      and licertupper a x0 x1 x2
        ratlt x1 x2 b c $.
$}

$(
##############################################################################
  Rational pointwise estimates for PNT and the von Koch RH criterion
##############################################################################
$)

$(
  pntat X E D means

    | pi(X) log(X) / X - 1 | < E/D,

  certified using rational lower and upper cuts for log(X).  The definition
  contains only cross-multiplied natural-number inequalities.
$)
${
  $d x0 a $. $d x0 b $. $d x0 c $.
  $d x1 a $. $d x1 b $. $d x1 c $.
  $d x2 a $. $d x2 b $. $d x2 c $.
  $d x3 a $. $d x3 b $. $d x3 c $.
  $d x4 a $. $d x4 b $. $d x4 c $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $. $d x0 x4 $.
  $d x1 x2 $. $d x1 x3 $. $d x1 x4 $.
  $d x2 x3 $. $d x2 x4 $. $d x3 x4 $.
  df-pntat $a |- iff pntat a b c
    exists x0 exists x1 exists x2 exists x3 exists x4
      and < 0 b
        and < 0 c
          and primecount a x0
            and lnlower a x1 x2
              and lnupper a x3 x4
                and < * c * x0 x3
                      * + c b * a x4
                  < * c * a x2
                    + * c * x0 x1
                      * b * a x2 $.
$}

$(
  The next two predicates remove square roots by squaring nonnegative rational
  inequalities.  Respectively they certify

    max(N - A/B, 0) <= C sqrt(X) G/H
    max(A/B - N, 0) <= C sqrt(X) G/H.
$)
${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $.
  $d x0 e $. $d x0 f $. $d x0 g $.
  df-natminusrootle $a |- iff natminusrootle a b c d e f g
    and < 0 c
      and < 0 g
        or le * a c b
          exists x0
            and = * a c + b x0
              le * * x0 x0 * g g
                * * * d d e
                  * * f f * c c $.
$}

${
  $d x0 a $. $d x0 b $. $d x0 c $. $d x0 d $.
  $d x0 e $. $d x0 f $. $d x0 g $.
  df-ratminusnatrootle $a |- iff ratminusnatrootle a b c d e f g
    and < 0 c
      and < 0 g
        or le b * a c
          exists x0
            and = b + * a c x0
              le * * x0 x0 * g g
                * * * d d e
                  * * f f * c c $.
$}

$(
  rhat X C is the rational-certificate form of

    | pi(X) - Li(X) | <= C sqrt(X) log(X).

  A lower Li certificate handles pi(X)-Li(X), an upper Li certificate handles
  Li(X)-pi(X), and a lower log certificate makes both inequalities entirely
  rational.  Existentially tightening the cuts recovers the usual bound.
$)
${
  $d x0 a $. $d x0 b $.
  $d x1 a $. $d x1 b $.
  $d x2 a $. $d x2 b $.
  $d x3 a $. $d x3 b $.
  $d x4 a $. $d x4 b $.
  $d x5 a $. $d x5 b $.
  $d x6 a $. $d x6 b $.
  $d x0 x1 $. $d x0 x2 $. $d x0 x3 $. $d x0 x4 $. $d x0 x5 $. $d x0 x6 $.
  $d x1 x2 $. $d x1 x3 $. $d x1 x4 $. $d x1 x5 $. $d x1 x6 $.
  $d x2 x3 $. $d x2 x4 $. $d x2 x5 $. $d x2 x6 $.
  $d x3 x4 $. $d x3 x5 $. $d x3 x6 $.
  $d x4 x5 $. $d x4 x6 $. $d x5 x6 $.
  df-rhat $a |- iff rhat a b
    exists x0 exists x1 exists x2 exists x3 exists x4 exists x5 exists x6
      and primecount a x0
        and lilower a x1 x2
          and liupper a x3 x4
            and lnlower a x5 x6
              and natminusrootle x0 x1 x2 b a x5 x6
                ratminusnatrootle x0 x3 x4 b a x5 x6 $.
$}

$(
##############################################################################
  Named formulas

  These declarations have type "statement".  In particular, none has a
  conclusion beginning with "|-", so no conjecture is silently added as an
  axiom or as an already proved theorem.
##############################################################################
$)

$(
  Fermat's Last Theorem:
  for exponent N > 2 and positive A,B,C, A^N + B^N is not C^N.
$)
flt-statement $a statement
  forall x0 forall x1 forall x2 forall x3
    forall x4 forall x5 forall x6
      implies
        and < S S 0 x0
          and < 0 x1
            and < 0 x2
              and < 0 x3
                and pow x1 x0 x4
                  and pow x2 x0 x5
                    pow x3 x0 x6
        not = + x4 x5 x6 $.

$(
  Strong Goldbach conjecture:
  every even N > 2 is the sum of two primes.
$)
goldbach-statement $a statement
  forall x0
    implies and < S S 0 x0 even x0
      exists x1 exists x2
        and prime x1
          and prime x2
            = x0 + x1 x2 $.

$(
  Prime Number Theorem without limit or asymptotic notation:
  for every positive rational E/D there is N >= 2 such that every X >= N
  satisfies |pi(X) log(X)/X - 1| < E/D.
$)
pnt-statement $a statement
  forall x0 forall x1
    implies and < 0 x0 < 0 x1
      exists x2
        and le S S 0 x2
          forall x3
            implies le x2 x3
              pntat x3 x0 x1 $.

$(
  von Koch's equivalent form of the Riemann Hypothesis, with Big-O expanded:
  there are natural C > 0 and N >= 2 such that for every X >= N,
  |pi(X)-Li(X)| <= C sqrt(X) log(X).  rhat eliminates sqrt and all real
  quantities using squared natural inequalities and rational cuts.
$)
riemann-von-koch-statement $a statement
  exists x0 exists x1
    and < 0 x0
      and le S S 0 x1
        forall x2
          implies le x1 x2
            rhat x2 x0 $.
