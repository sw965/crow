package tensor3ds

import (
	"math/rand"
	"github.com/sw965/crow/blas32/tensor/3d"
)

type Generals []tensor3d.General

func NewZerosLike(gens Generals) Generals {
	zeros := make(Generals, len(gens))
	for i, gen := range gens {
		zeros[i] = tensor3d.NewZerosLike(gen)
	}
	return zeros
}

func NewRademacherLike(gens Generals, rng *rand.Rand) Generals {
	rad := make(Generals, len(gens))
	for i, gen := range gens {
		rad[i] = tensor3d.NewRademacherLike(gen, rng)
	}
	return rad
}

func Clone(gens Generals) Generals {
	clone := make(Generals, len(gens))
	for i, gen := range gens {
		clone[i] = gen.Clone()
	}
	return clone
}