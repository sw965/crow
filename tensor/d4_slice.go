package tensor

import (
	"math/rand"
)

type D4Slice []D4

func (ds D4Slice) NewZerosLike() D4Slice {
	zeros := make(D4Slice, len(ds))
	for i, d := range ds {
		zeros[i] = d.NewZerosLike()
	}
	return zeros
}

func (ds D4Slice) NewOnesLike() D4Slice {
	ones := make(D4Slice, len(ds))
	for i, d := range ds {
		ones[i] = d.NewOnesLike()
	}
	return ones
}

func (ds D4Slice) NewRademacherLike(rng *rand.Rand) D4Slice {
	rad := make(D4Slice, len(ds))
	for i, d := range ds {
		rad[i] = d.NewRademacherLike(rng)
	}
	return rad
}

func (ds D4Slice) Clone() D4Slice {
	clone := make(D4Slice, len(ds))
	for i, d := range ds {
		clone[i] = d.Clone()
	}
	return clone
}