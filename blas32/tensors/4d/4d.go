package tensor4ds

import (
	"math/rand"
	"github.com/sw965/crow/blas32/tensor/4d"
)

type Generals []tensor4d.General

func NewZerosLike(gens Generals) Generals {
	zeros := make(Generals, len(gens))
	for i, gen := range gens {
		zeros[i] = tensor4d.NewZerosLike(gen)
	}
	return zeros
}

func NewRademacherLike(gens Generals, rng *rand.Rand) Generals {
	rad := make(Generals, len(gens))
	for i, gen := range gens {
		rad[i] = tensor4d.NewRademacherLike(gen, rng)
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