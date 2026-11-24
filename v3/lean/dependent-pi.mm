$( Dependent functions, using vectors of natural numbers indexed by length.
   Pi A B represents (x : A) -> B x. The family B takes the index as
   an argument, so this notation needs no named object-language binder. $)

$c expr |- : Type family Pi app Nat VecNat 0 1 $.
$v A B f x $.
sA $f expr A $.  sB $f expr B $.  sf $f expr f $.  sx $f expr x $.

$( Raw expressions. Mathematical typing is a separate judgment starting
   with the turnstile. $)
sNat $a expr Nat $.  sVec $a expr VecNat $.
s0 $a expr 0 $.      s1 $a expr 1 $.
sPi $a expr Pi A B $.  sApp $a expr app f x $.

$( VecNat sends a natural number n to the type of length-n vectors. $)
nat-type   $a |- Nat : Type $.
vec-family $a |- family VecNat Nat $.
zero-nat   $a |- 0 : Nat $.
one-nat    $a |- 1 : Nat $.

${
  form-domain $e |- A : Type $.
  form-family $e |- family B A $.
  pi-form     $a |- Pi A B : Type $.
$}

vec-pi $p |- Pi Nat VecNat : Type $=
  sNat sVec nat-type vec-family pi-form $.

${
  use-fun $e |- f : Pi A B $.
  use-arg $e |- x : A $.
  pi-app  $a |- app f x : app B x $.
$}

$( Both theorems use the same local function hypothesis. $)
${
  hf $e |- f : Pi Nat VecNat $.
  at-zero $p |- app f 0 : app VecNat 0 $=
    sNat sVec sf s0 hf zero-nat pi-app $.
  at-one $p |- app f 1 : app VecNat 1 $=
    sNat sVec sf s1 hf one-nat pi-app $.
$}
