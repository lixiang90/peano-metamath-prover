prelude
import Init.Prelude

namespace V3Probe

universe u v

/-- The natural-number interpretation of frozen PA case p0004. -/
theorem natOneMul (n : Nat) : Eq n (Nat.mul (Nat.succ Nat.zero) n) :=
  Nat.rec (motive := fun k => Eq k (Nat.mul (Nat.succ Nat.zero) k))
    (Eq.refl Nat.zero) (fun _ ih => congrArg Nat.succ ih) n

theorem succAdd (a b : Nat) : Eq (Nat.add (Nat.succ a) b) (Nat.succ (Nat.add a b)) :=
  Nat.rec (motive := fun k => Eq (Nat.add (Nat.succ a) k) (Nat.succ (Nat.add a k)))
    (Eq.refl (Nat.succ a)) (fun _ ih => congrArg Nat.succ ih) b

/-- The natural-number statement of Fermat's Last Theorem. -/
def FermatStatement : Prop :=
  ∀ (n a b c : Nat), Nat.lt 2 n →
    Eq (Nat.add (Nat.pow a n) (Nat.pow b n)) (Nat.pow c n) →
    Or (Eq a 0) (Or (Eq b 0) (Eq c 0))

def power {G : Type u} (unit : G) (mul : G → G → G) (a : G) : Nat → G
  | Nat.zero => unit
  | Nat.succ n => mul (power unit mul a n) a

/-- A map preserving the unit and multiplication preserves recursively defined powers. -/
theorem mapPower {G : Type u} {H : Type v} (unitG : G) (unitH : H)
    (mulG : G → G → G) (mulH : H → H → H) (f : G → H)
    (mapUnit : Eq (f unitG) unitH)
    (mapMul : ∀ x y, Eq (f (mulG x y)) (mulH (f x) (f y)))
    (a : G) (n : Nat) :
    Eq (f (power unitG mulG a n)) (power unitH mulH (f a) n) :=
  Nat.rec (motive := fun k => Eq (f (power unitG mulG a k)) (power unitH mulH (f a) k))
    mapUnit
    (fun k ih => Eq.trans (mapMul (power unitG mulG a k) a)
      (congrArg (fun z => mulH z (f a)) ih))
    n

/-- A proof of divisibility supplies the factor multiplying `d` on the right. -/
inductive Divides {R : Type u} (mul : R → R → R) (d : R) : R → Prop where
  | intro (x : R) : Divides mul d (mul d x)

theorem dividesAdd {R : Type u} (add mul : R → R → R)
    (distrib : ∀ d x y, Eq (mul d (add x y)) (add (mul d x) (mul d y)))
    {d a b : R} (ha : Divides mul d a) (hb : Divides mul d b) :
    Divides mul d (add a b) :=
  match ha, hb with
  | .intro x, .intro y =>
    Eq.ndrec (motive := fun t => Divides mul d t)
      (Divides.intro (add x y)) (distrib d x y)

/-- Both dimensions determine the type of a matrix. -/
abbrev Matrix (R : Type u) (m n : Nat) := Fin m → Fin n → R

def transpose {R : Type u} {m n : Nat} (a : Matrix R m n) : Matrix R n m :=
  fun j i => a i j

theorem transposeTwice {R : Type u} {m n : Nat} (a : Matrix R m n) :
    Eq (transpose (transpose a)) a := rfl

#print axioms natOneMul
#print axioms succAdd
#print axioms mapPower
#print axioms dividesAdd
#print axioms transposeTwice

end V3Probe
