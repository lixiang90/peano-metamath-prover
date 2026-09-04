$(
  GENERATED FILE: edit formal/pa-plus-definitions.json and regenerate.
  Every logical assertion below is an acyclic explicit definition.
  Named theorem formulas have type statement and assert no theorem.
$)

$[ peano-number-theory.mm $]

$( Fresh high-level predicate syntax. $)
$c natne positive ge gt between natdiff minrel maxrel properdivides composite coprime $.
$c gcdrel lcmrel congruent remainder square cube sumtwosquares sumfoursquares $.
$c pythagorean pellsolution quadraticresidue squarefree valuation perfectpower $.
$c multiplicativeorder mersenne fermatnumber seqat seqsum seqprod factorial fibonacci $.
$c binomial arithterm geomterm arithsum geomseries divisorcount divisorsum eulerphi $.
$c perfectnumber abundantnumber deficientnumber inteq intle intlt intadd intmul intneg $.
$c intabs qvalid qeq qle qlt qadd qmul qneg qabs ratbetween ratabsle ratabslt sqrtlower $.
$c sqrtupper sqrtinterval nthrootlower nthrootupper nthrootexact $.
wff_natne $a wff natne a b $.
wff_positive $a wff positive a $.
wff_ge $a wff ge a b $.
wff_gt $a wff gt a b $.
wff_between $a wff between a b c $.
wff_natdiff $a wff natdiff a b c $.
wff_minrel $a wff minrel a b c $.
wff_maxrel $a wff maxrel a b c $.
wff_properdivides $a wff properdivides a b $.
wff_composite $a wff composite a $.
wff_coprime $a wff coprime a b $.
wff_gcdrel $a wff gcdrel a b c $.
wff_lcmrel $a wff lcmrel a b c $.
wff_congruent $a wff congruent a b c $.
wff_remainder $a wff remainder a b c $.
wff_square $a wff square a $.
wff_cube $a wff cube a $.
wff_sumtwosquares $a wff sumtwosquares a $.
wff_sumfoursquares $a wff sumfoursquares a $.
wff_pythagorean $a wff pythagorean a b c $.
wff_pellsolution $a wff pellsolution a b c $.
wff_quadraticresidue $a wff quadraticresidue a b $.
wff_squarefree $a wff squarefree a $.
wff_valuation $a wff valuation a b c $.
wff_perfectpower $a wff perfectpower a $.
wff_multiplicativeorder $a wff multiplicativeorder a b c $.
wff_mersenne $a wff mersenne a b $.
wff_fermatnumber $a wff fermatnumber a b $.
wff_seqat $a wff seqat a b c d $.
wff_seqsum $a wff seqsum a b c d $.
wff_seqprod $a wff seqprod a b c d $.
wff_factorial $a wff factorial a b $.
wff_fibonacci $a wff fibonacci a b $.
wff_binomial $a wff binomial a b c $.
wff_arithterm $a wff arithterm a b c d $.
wff_geomterm $a wff geomterm a b c d $.
wff_arithsum $a wff arithsum a b c d $.
wff_geomseries $a wff geomseries a b c d $.
wff_divisorcount $a wff divisorcount a b $.
wff_divisorsum $a wff divisorsum a b $.
wff_eulerphi $a wff eulerphi a b $.
wff_perfectnumber $a wff perfectnumber a $.
wff_abundantnumber $a wff abundantnumber a $.
wff_deficientnumber $a wff deficientnumber a $.
wff_inteq $a wff inteq a b c d $.
wff_intle $a wff intle a b c d $.
wff_intlt $a wff intlt a b c d $.
wff_intadd $a wff intadd a b c d e f $.
wff_intmul $a wff intmul a b c d e f $.
wff_intneg $a wff intneg a b c d $.
wff_intabs $a wff intabs a b c $.
wff_qvalid $a wff qvalid a b c $.
wff_qeq $a wff qeq a b c d e f $.
wff_qle $a wff qle a b c d e f $.
wff_qlt $a wff qlt a b c d e f $.
wff_qadd $a wff qadd a b c d e f g h i $.
wff_qmul $a wff qmul a b c d e f g h i $.
wff_qneg $a wff qneg a b c d e f $.
wff_qabs $a wff qabs a b c d e $.
wff_ratbetween $a wff ratbetween a b c d e f $.
wff_ratabsle $a wff ratabsle a b c d e f $.
wff_ratabslt $a wff ratabslt a b c d e f $.
wff_sqrtlower $a wff sqrtlower a b c $.
wff_sqrtupper $a wff sqrtupper a b c $.
wff_sqrtinterval $a wff sqrtinterval a b c d e $.
wff_nthrootlower $a wff nthrootlower a b c d $.
wff_nthrootupper $a wff nthrootupper a b c d $.
wff_nthrootexact $a wff nthrootexact a b c d $.

$( Conservative explicit definitions. $)

df-natne $a |- iff natne a b not = a b $.

df-positive $a |- iff positive a < 0 a $.

df-ge $a |- iff ge a b le b a $.

df-gt $a |- iff gt a b < b a $.

df-between $a |- iff between a b c and le a b le b c $.

df-natdiff $a |- iff natdiff a b c = a + b c $.

df-minrel $a |- iff minrel a b c or and le a b = c a and le b a = c b $.

df-maxrel $a |- iff maxrel a b c or and le a b = c b and le b a = c a $.

df-properdivides $a |- iff properdivides a b and divides a b < a b $.

${
  $d x0 a $.
  $d x1 a $.
  $d x0 x1 $.
  df-composite $a |- iff composite a exists x0 exists x1 and < S 0 x0 and < S 0 x1 = a * x0 x1 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  df-coprime $a |- iff coprime a b forall x0 implies and divides x0 a divides x0 b = x0 S 0 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  df-gcdrel $a |- iff gcdrel a b c and divides c a and divides c b forall x0 implies and divides x0 a divides x0 b divides x0 c $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  df-lcmrel $a |- iff lcmrel a b c and divides a c and divides b c forall x0 implies and divides a x0 divides b x0 divides c x0 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x1 a $.
  $d x1 b $.
  $d x1 c $.
  $d x0 x1 $.
  df-congruent $a |- iff congruent a b c and positive c exists x0 exists x1 = + a * c x0 + b * c x1 $.
$}

df-remainder $a |- iff remainder a b c and < c b congruent a c b $.

${
  $d x0 a $.
  df-square $a |- iff square a exists x0 = a * x0 x0 $.
$}

${
  $d x0 a $.
  df-cube $a |- iff cube a exists x0 = a * * x0 x0 x0 $.
$}

${
  $d x0 a $.
  $d x1 a $.
  $d x0 x1 $.
  df-sumtwosquares $a |- iff sumtwosquares a exists x0 exists x1 = a + * x0 x0 * x1 x1 $.
$}

${
  $d x0 a $.
  $d x1 a $.
  $d x2 a $.
  $d x3 a $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x0 x3 $.
  $d x1 x2 $.
  $d x1 x3 $.
  $d x2 x3 $.
  df-sumfoursquares $a |- iff sumfoursquares a exists x0 exists x1 exists x2 exists x3 = a + * x0 x0 + * x1 x1 + * x2 x2 * x3 x3 $.
$}

df-pythagorean $a |- iff pythagorean a b c = + * a a * b b * c c $.

df-pellsolution $a |- iff pellsolution a b c = * b b + S 0 * a * c c $.

${
  $d x0 a $.
  $d x0 b $.
  df-quadraticresidue $a |- iff quadraticresidue a b and positive b exists x0 congruent * x0 x0 a b $.
$}

${
  $d x0 a $.
  df-squarefree $a |- iff squarefree a forall x0 implies divides * x0 x0 a = x0 S 0 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x1 a $.
  $d x1 b $.
  $d x1 c $.
  $d x0 x1 $.
  df-valuation $a |- iff valuation a b c and prime a and positive b exists x0 exists x1 and pow a c x0 and pow a S c x1 and divides x0 b not divides x1 b $.
$}

${
  $d x0 a $.
  $d x1 a $.
  $d x0 x1 $.
  df-perfectpower $a |- iff perfectpower a exists x0 exists x1 and < S 0 x0 and < S 0 x1 pow x0 x1 a $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x1 a $.
  $d x1 b $.
  $d x1 c $.
  $d x2 a $.
  $d x2 b $.
  $d x2 c $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x1 x2 $.
  df-multiplicativeorder $a |- iff multiplicativeorder a b c and < S 0 b and coprime a b and < 0 c and exists x0 and pow a c x0 congruent x0 S 0 b forall x1 implies and < 0 x1 < x1 c exists x2 and pow a x1 x2 not congruent x2 S 0 b $.
$}

${
  $d x0 a $.
  $d x0 b $.
  df-mersenne $a |- iff mersenne a b exists x0 and pow S S 0 b x0 = x0 S a $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x1 a $.
  $d x1 b $.
  $d x0 x1 $.
  df-fermatnumber $a |- iff fermatnumber a b exists x0 exists x1 and pow S S 0 b x0 and pow S S 0 x0 x1 = a S x1 $.
$}

df-seqat $a |- iff seqat a b c d beta a b c d $.

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x0 d $.
  $d x1 a $.
  $d x1 b $.
  $d x1 c $.
  $d x1 d $.
  $d x2 a $.
  $d x2 b $.
  $d x2 c $.
  $d x2 d $.
  $d x3 a $.
  $d x3 b $.
  $d x3 c $.
  $d x3 d $.
  $d x4 a $.
  $d x4 b $.
  $d x4 c $.
  $d x4 d $.
  $d x5 a $.
  $d x5 b $.
  $d x5 c $.
  $d x5 d $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x0 x3 $.
  $d x0 x4 $.
  $d x0 x5 $.
  $d x1 x2 $.
  $d x1 x3 $.
  $d x1 x4 $.
  $d x1 x5 $.
  $d x2 x3 $.
  $d x2 x4 $.
  $d x2 x5 $.
  $d x3 x4 $.
  $d x3 x5 $.
  $d x4 x5 $.
  df-seqsum $a |- iff seqsum a b c d exists x0 exists x1 and seqat x0 x1 0 0 and seqat x0 x1 c d forall x2 implies < x2 c exists x3 exists x4 exists x5 and seqat a b x2 x3 and seqat x0 x1 x2 x4 and seqat x0 x1 S x2 x5 = x5 + x4 x3 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x0 d $.
  $d x1 a $.
  $d x1 b $.
  $d x1 c $.
  $d x1 d $.
  $d x2 a $.
  $d x2 b $.
  $d x2 c $.
  $d x2 d $.
  $d x3 a $.
  $d x3 b $.
  $d x3 c $.
  $d x3 d $.
  $d x4 a $.
  $d x4 b $.
  $d x4 c $.
  $d x4 d $.
  $d x5 a $.
  $d x5 b $.
  $d x5 c $.
  $d x5 d $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x0 x3 $.
  $d x0 x4 $.
  $d x0 x5 $.
  $d x1 x2 $.
  $d x1 x3 $.
  $d x1 x4 $.
  $d x1 x5 $.
  $d x2 x3 $.
  $d x2 x4 $.
  $d x2 x5 $.
  $d x3 x4 $.
  $d x3 x5 $.
  $d x4 x5 $.
  df-seqprod $a |- iff seqprod a b c d exists x0 exists x1 and seqat x0 x1 0 S 0 and seqat x0 x1 c d forall x2 implies < x2 c exists x3 exists x4 exists x5 and seqat a b x2 x3 and seqat x0 x1 x2 x4 and seqat x0 x1 S x2 x5 = x5 * x4 x3 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x1 a $.
  $d x1 b $.
  $d x2 a $.
  $d x2 b $.
  $d x3 a $.
  $d x3 b $.
  $d x4 a $.
  $d x4 b $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x0 x3 $.
  $d x0 x4 $.
  $d x1 x2 $.
  $d x1 x3 $.
  $d x1 x4 $.
  $d x2 x3 $.
  $d x2 x4 $.
  $d x3 x4 $.
  df-factorial $a |- iff factorial a b exists x0 exists x1 and seqat x0 x1 0 S 0 and seqat x0 x1 a b forall x2 implies < x2 a exists x3 exists x4 and seqat x0 x1 x2 x3 and seqat x0 x1 S x2 x4 = x4 * x3 S x2 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x1 a $.
  $d x1 b $.
  $d x2 a $.
  $d x2 b $.
  $d x3 a $.
  $d x3 b $.
  $d x4 a $.
  $d x4 b $.
  $d x5 a $.
  $d x5 b $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x0 x3 $.
  $d x0 x4 $.
  $d x0 x5 $.
  $d x1 x2 $.
  $d x1 x3 $.
  $d x1 x4 $.
  $d x1 x5 $.
  $d x2 x3 $.
  $d x2 x4 $.
  $d x2 x5 $.
  $d x3 x4 $.
  $d x3 x5 $.
  $d x4 x5 $.
  df-fibonacci $a |- iff fibonacci a b exists x0 exists x1 and seqat x0 x1 0 0 and seqat x0 x1 S 0 S 0 and seqat x0 x1 a b forall x2 implies < S x2 a exists x3 exists x4 exists x5 and seqat x0 x1 x2 x3 and seqat x0 x1 S x2 x4 and seqat x0 x1 S S x2 x5 = x5 + x3 x4 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x1 a $.
  $d x1 b $.
  $d x1 c $.
  $d x2 a $.
  $d x2 b $.
  $d x2 c $.
  $d x3 a $.
  $d x3 b $.
  $d x3 c $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x0 x3 $.
  $d x1 x2 $.
  $d x1 x3 $.
  $d x2 x3 $.
  df-binomial $a |- iff binomial a b c or exists x0 exists x1 exists x2 exists x3 and natdiff a b x0 and factorial a x1 and factorial b x2 and factorial x0 x3 = x1 * c * x2 x3 and < a b = c 0 $.
$}

df-arithterm $a |- iff arithterm a b c d = d + a * b c $.

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x0 d $.
  df-geomterm $a |- iff geomterm a b c d exists x0 and pow b c x0 = d * a x0 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x0 d $.
  $d x1 a $.
  $d x1 b $.
  $d x1 c $.
  $d x1 d $.
  $d x2 a $.
  $d x2 b $.
  $d x2 c $.
  $d x2 d $.
  $d x3 a $.
  $d x3 b $.
  $d x3 c $.
  $d x3 d $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x0 x3 $.
  $d x1 x2 $.
  $d x1 x3 $.
  $d x2 x3 $.
  df-arithsum $a |- iff arithsum a b c d exists x0 exists x1 and seqsum x0 x1 c d forall x2 implies < x2 c exists x3 and seqat x0 x1 x2 x3 arithterm a b x2 x3 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x0 d $.
  $d x1 a $.
  $d x1 b $.
  $d x1 c $.
  $d x1 d $.
  $d x2 a $.
  $d x2 b $.
  $d x2 c $.
  $d x2 d $.
  $d x3 a $.
  $d x3 b $.
  $d x3 c $.
  $d x3 d $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x0 x3 $.
  $d x1 x2 $.
  $d x1 x3 $.
  $d x2 x3 $.
  df-geomseries $a |- iff geomseries a b c d exists x0 exists x1 and seqsum x0 x1 c d forall x2 implies < x2 c exists x3 and seqat x0 x1 x2 x3 geomterm a b x2 x3 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x1 a $.
  $d x1 b $.
  $d x2 a $.
  $d x2 b $.
  $d x3 a $.
  $d x3 b $.
  $d x4 a $.
  $d x4 b $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x0 x3 $.
  $d x0 x4 $.
  $d x1 x2 $.
  $d x1 x3 $.
  $d x1 x4 $.
  $d x2 x3 $.
  $d x2 x4 $.
  $d x3 x4 $.
  df-divisorcount $a |- iff divisorcount a b and positive a exists x0 exists x1 and seqat x0 x1 0 0 and seqat x0 x1 a b forall x2 implies < x2 a exists x3 exists x4 and seqat x0 x1 x2 x3 and seqat x0 x1 S x2 x4 and implies divides S x2 a = x4 S x3 implies not divides S x2 a = x4 x3 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x1 a $.
  $d x1 b $.
  $d x2 a $.
  $d x2 b $.
  $d x3 a $.
  $d x3 b $.
  $d x4 a $.
  $d x4 b $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x0 x3 $.
  $d x0 x4 $.
  $d x1 x2 $.
  $d x1 x3 $.
  $d x1 x4 $.
  $d x2 x3 $.
  $d x2 x4 $.
  $d x3 x4 $.
  df-divisorsum $a |- iff divisorsum a b and positive a exists x0 exists x1 and seqat x0 x1 0 0 and seqat x0 x1 a b forall x2 implies < x2 a exists x3 exists x4 and seqat x0 x1 x2 x3 and seqat x0 x1 S x2 x4 and implies divides S x2 a = x4 + x3 S x2 implies not divides S x2 a = x4 x3 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x1 a $.
  $d x1 b $.
  $d x2 a $.
  $d x2 b $.
  $d x3 a $.
  $d x3 b $.
  $d x4 a $.
  $d x4 b $.
  $d x0 x1 $.
  $d x0 x2 $.
  $d x0 x3 $.
  $d x0 x4 $.
  $d x1 x2 $.
  $d x1 x3 $.
  $d x1 x4 $.
  $d x2 x3 $.
  $d x2 x4 $.
  $d x3 x4 $.
  df-eulerphi $a |- iff eulerphi a b and positive a exists x0 exists x1 and seqat x0 x1 0 0 and seqat x0 x1 a b forall x2 implies < x2 a exists x3 exists x4 and seqat x0 x1 x2 x3 and seqat x0 x1 S x2 x4 and implies coprime S x2 a = x4 S x3 implies not coprime S x2 a = x4 x3 $.
$}

${
  $d x0 a $.
  df-perfectnumber $a |- iff perfectnumber a and positive a exists x0 and divisorsum a x0 = x0 + a a $.
$}

${
  $d x0 a $.
  df-abundantnumber $a |- iff abundantnumber a and positive a exists x0 and divisorsum a x0 < + a a x0 $.
$}

${
  $d x0 a $.
  df-deficientnumber $a |- iff deficientnumber a and positive a exists x0 and divisorsum a x0 < x0 + a a $.
$}

df-inteq $a |- iff inteq a b c d = + a d + c b $.

df-intle $a |- iff intle a b c d le + a d + c b $.

df-intlt $a |- iff intlt a b c d < + a d + c b $.

df-intadd $a |- iff intadd a b c d e f inteq e f + a c + b d $.

df-intmul $a |- iff intmul a b c d e f inteq e f + * a c * b d + * a d * b c $.

df-intneg $a |- iff intneg a b c d inteq c d b a $.

df-intabs $a |- iff intabs a b c or = a + b c = b + a c $.

df-qvalid $a |- iff qvalid a b c positive c $.

df-qeq $a |- iff qeq a b c d e f and positive c and positive f inteq * a f * b f * d c * e c $.

df-qle $a |- iff qle a b c d e f and positive c and positive f intle * a f * b f * d c * e c $.

df-qlt $a |- iff qlt a b c d e f and positive c and positive f intlt * a f * b f * d c * e c $.

df-qadd $a |- iff qadd a b c d e f g h i and qvalid a b c and qvalid d e f and qvalid g h i qeq g h i + * a f * d c + * b f * e c * c f $.

df-qmul $a |- iff qmul a b c d e f g h i and qvalid a b c and qvalid d e f and qvalid g h i qeq g h i + * a d * b e + * a e * b d * c f $.

df-qneg $a |- iff qneg a b c d e f qeq d e f b a c $.

df-qabs $a |- iff qabs a b c d e and positive e or and le b a qeq a b c d 0 e and le a b qeq b a c d 0 e $.

df-ratbetween $a |- iff ratbetween a b c d e f and ratle c d a b ratle a b e f $.

df-ratabsle $a |- iff ratabsle a b c d e f and positive b and positive d and positive f and le * f * a d + * f * c b * e * b d le * f * c b + * f * a d * e * b d $.

df-ratabslt $a |- iff ratabslt a b c d e f and positive b and positive d and positive f and < * f * a d + * f * c b * e * b d < * f * c b + * f * a d * e * b d $.

df-sqrtlower $a |- iff sqrtlower a b c and positive c < * b b * a * c c $.

df-sqrtupper $a |- iff sqrtupper a b c and positive c < * a * c c * b b $.

df-sqrtinterval $a |- iff sqrtinterval a b c d e and sqrtlower a b c sqrtupper a d e $.

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x0 d $.
  $d x1 a $.
  $d x1 b $.
  $d x1 c $.
  $d x1 d $.
  $d x0 x1 $.
  df-nthrootlower $a |- iff nthrootlower a b c d and positive b and positive d exists x0 exists x1 and pow c b x0 and pow d b x1 < x0 * a x1 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x0 d $.
  $d x1 a $.
  $d x1 b $.
  $d x1 c $.
  $d x1 d $.
  $d x0 x1 $.
  df-nthrootupper $a |- iff nthrootupper a b c d and positive b and positive d exists x0 exists x1 and pow c b x0 and pow d b x1 < * a x1 x0 $.
$}

${
  $d x0 a $.
  $d x0 b $.
  $d x0 c $.
  $d x0 d $.
  $d x1 a $.
  $d x1 b $.
  $d x1 c $.
  $d x1 d $.
  $d x0 x1 $.
  df-nthrootexact $a |- iff nthrootexact a b c d and positive b and positive d exists x0 exists x1 and pow c b x0 and pow d b x1 = x0 * a x1 $.
$}

$( Closed target formulas only; these are not logical assertions. $)

$( There are arbitrarily large primes. $)
euclid-primes-statement $a statement forall x0 exists x1 and < x0 x1 prime x1 $.

$( Division with quotient and a smaller remainder. $)
division-algorithm-statement $a statement forall x0 forall x1 implies positive x1 exists x2 exists x3 and = x0 + * x1 x2 x3 < x3 x1 $.

$( Every pair of naturals has a greatest common divisor. $)
gcd-existence-statement $a statement forall x0 forall x1 exists x2 gcdrel x0 x1 x2 $.

$( Every pair of positive naturals has a least common multiple. $)
lcm-existence-statement $a statement forall x0 forall x1 exists x2 lcmrel x0 x1 x2 $.

$( The product of gcd and lcm equals the product of the inputs. $)
gcd-lcm-product-statement $a statement forall x0 forall x1 forall x2 forall x3 implies and gcdrel x0 x1 x2 lcmrel x0 x1 x3 = * x2 x3 * x0 x1 $.

$( Every natural above one has a prime divisor. $)
prime-divisor-statement $a statement forall x0 implies < S 0 x0 exists x1 and prime x1 divides x1 x0 $.

$( Coprime naturals admit a signed Bezout identity. $)
bezout-coprime-statement $a statement forall x0 forall x1 implies coprime x0 x1 exists x2 exists x3 exists x4 exists x5 inteq + * x0 x2 * x1 x4 + * x0 x3 * x1 x5 S 0 0 $.

$( Chinese remainder theorem for two coprime moduli. $)
crt-two-moduli-statement $a statement forall x0 forall x1 forall x2 forall x3 implies and positive x2 and positive x3 coprime x2 x3 exists x4 and congruent x4 x0 x2 congruent x4 x1 x3 $.

$( Fermat's little theorem in a^p congruent a form. $)
fermat-little-statement $a statement forall x0 forall x1 implies prime x1 exists x2 and pow x0 x1 x2 congruent x2 x0 x1 $.

$( Euler's theorem for coprime base and modulus. $)
euler-theorem-statement $a statement forall x0 forall x1 forall x2 implies and positive x1 and coprime x0 x1 eulerphi x1 x2 exists x3 and pow x0 x2 x3 congruent x3 S 0 x1 $.

$( Wilson's theorem: (p-1)! is congruent to -1 modulo p. $)
wilson-statement $a statement forall x0 implies prime x0 exists x1 exists x2 and natdiff x0 S 0 x1 and factorial x1 x2 congruent x2 x1 x0 $.

$( A prime congruent to one modulo four is a sum of two squares. $)
fermat-two-squares-statement $a statement forall x0 implies and prime x0 congruent x0 S 0 S S S S 0 sumtwosquares x0 $.

$( Every natural is a sum of four squares. $)
lagrange-four-squares-statement $a statement forall x0 sumfoursquares x0 $.

$( Every positive nonsquare parameter has a nontrivial Pell solution. $)
pell-existence-statement $a statement forall x0 implies and positive x0 not square x0 exists x1 exists x2 and positive x2 pellsolution x0 x1 x2 $.

$( Every natural has a squarefree-times-square decomposition. $)
squarefree-divisor-statement $a statement forall x0 exists x1 exists x2 and squarefree x1 and square x2 = x0 * x1 x2 $.

$( Multiplicative order divides Euler's totient. $)
order-divides-phi-statement $a statement forall x0 forall x1 forall x2 forall x3 implies and multiplicativeorder x0 x1 x2 eulerphi x1 x3 divides x2 x3 $.

$( The divisor count is odd exactly for squares. $)
divisorcount-square-statement $a statement forall x0 forall x1 implies and positive x0 divisorcount x0 x1 iff even x1 not square x0 $.

$( Every prime is deficient. $)
prime-deficient-statement $a statement forall x0 implies prime x0 deficientnumber x0 $.

$( The Euclid construction from a Mersenne prime yields an even perfect number. $)
euclid-even-perfect-statement $a statement forall x0 forall x1 implies and prime x1 mersenne x1 x0 exists x2 exists x3 exists x4 and natdiff x0 S 0 x3 and pow S S 0 x3 x4 and = x2 * x4 x1 perfectnumber x2 $.

$( Zero factorial equals one. $)
factorial-zero-statement $a statement factorial 0 S 0 $.

$( Factorial recurrence. $)
factorial-step-statement $a statement forall x0 forall x1 forall x2 implies and factorial x0 x1 factorial S x0 x2 = x2 * S x0 x1 $.

$( Fibonacci recurrence. $)
fibonacci-step-statement $a statement forall x0 forall x1 forall x2 forall x3 implies and fibonacci x0 x1 and fibonacci S x0 x2 fibonacci S S x0 x3 = x3 + x1 x2 $.

$( Binomial coefficient symmetry. $)
binomial-symmetry-statement $a statement forall x0 forall x1 forall x2 forall x3 forall x4 implies and natdiff x0 x1 x2 and binomial x0 x1 x3 binomial x0 x2 x4 = x3 x4 $.

$( Pascal's binomial recurrence. $)
pascal-statement $a statement forall x0 forall x1 forall x2 forall x3 forall x4 implies and positive x1 and binomial x0 x1 x2 and binomial x0 x3 x4 natdiff x1 S 0 x3 exists x5 and binomial S x0 x1 x5 = x5 + x2 x4 $.

$( Twice an arithmetic sum equals n times twice the start plus (n-1)d. $)
arithmetic-series-statement $a statement forall x0 forall x1 forall x2 forall x3 implies and positive x2 arithsum x0 x1 x2 x3 exists x4 and natdiff x2 S 0 x4 = * S S 0 x3 * x2 + * S S 0 x0 * x1 x4 $.

$( Cross-multiplied finite geometric series identity. $)
geometric-series-statement $a statement forall x0 forall x1 forall x2 forall x3 forall x4 implies and geomseries x0 x1 x2 x3 pow x1 x2 x4 = + * x3 x1 x0 + x3 * x0 x4 $.

$( Triangle inequality for integer-pair absolute value. $)
integer-triangle-statement $a statement forall x0 forall x1 forall x2 forall x3 forall x4 forall x5 forall x6 implies and intabs x0 x1 x4 and intabs x2 x3 x5 intabs + x0 x2 + x1 x3 x6 le x6 + x4 x5 $.

$( A signed rational lies strictly between any two distinct signed rationals. $)
rational-density-statement $a statement forall x0 forall x1 forall x2 forall x3 forall x4 forall x5 implies qlt x0 x1 x2 x3 x4 x5 exists x6 exists x7 exists x8 and qlt x0 x1 x2 x6 x7 x8 qlt x6 x7 x8 x3 x4 x5 $.

$( Triangle inequality for signed rationals. $)
rational-triangle-statement $a statement forall x0 forall x1 forall x2 forall x3 forall x4 forall x5 forall x6 forall x7 forall x8 forall x9 forall x10 forall x11 forall x12 forall x13 forall x14 implies and qadd x0 x1 x2 x3 x4 x5 x6 x7 x8 and qabs x0 x1 x2 x9 x10 and qabs x3 x4 x5 x11 x12 qabs x6 x7 x8 x13 x14 ratle x13 x14 + * x9 x12 * x11 x10 * x10 x12 $.

$( Archimedean bound in natural cross-multiplied form. $)
archimedean-statement $a statement forall x0 forall x1 implies positive x1 exists x2 < x0 * x1 x2 $.

$( Every positive square root is bracketed by rational endpoints of a fixed positive denominator. $)
sqrt-rational-cell-statement $a statement forall x0 forall x1 implies and positive x0 positive x1 exists x2 exists x3 sqrtinterval x0 x2 x1 x3 x1 $.

$( The square root of a prime has no reduced rational representation. $)
sqrt-prime-irrational-statement $a statement forall x0 forall x1 forall x2 implies and prime x0 and positive x2 coprime x1 x2 not = * x1 x1 * x0 * x2 x2 $.

$( A reduced rational whose positive power is a natural has denominator one. $)
nthroot-rational-criterion-statement $a statement forall x0 forall x1 forall x2 forall x3 implies and nthrootexact x0 x1 x2 x3 coprime x2 x3 = x3 S 0 $.

$( A strict square-root interval has its lower endpoint below its upper endpoint. $)
sqrt-interval-order-statement $a statement forall x0 forall x1 forall x2 forall x3 forall x4 implies sqrtinterval x0 x1 x2 x3 x4 ratlt x1 x2 x3 x4 $.

$( An absolute-error certificate remains valid under a larger tolerance. $)
absolute-error-monotone-statement $a statement forall x0 forall x1 forall x2 forall x3 forall x4 forall x5 forall x6 forall x7 implies and ratabsle x0 x1 x2 x3 x4 x5 ratle x4 x5 x6 x7 ratabsle x0 x1 x2 x3 x6 x7 $.
