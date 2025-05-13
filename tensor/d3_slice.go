package tensor

import (
	"math/rand"
)

type D3Slice []D3

func (ds D3Slice) NewZerosLike() D3Slice {
	zeros := make(D3Slice, len(ds))
	for i, d := range ds {
		zeros[i] = d.NewZerosLike()
	}
	return zeros
}

func (ds D3Slice) NewOnesLike() D3Slice {
	ones := make(D3Slice, len(ds))
	for i, d := range ds {
		ones[i] = d.NewOnesLike()
	}
	return ones
}

func (ds D3Slice) NewRademacherLike(rng *rand.Rand) D3Slice {
	rad := make(D3Slice, len(ds))
	for i, d := range ds {
		rad[i] = d.NewRademacherLike(rng)
	}
	return rad
}

func (ds D3Slice) Clone() D3Slice {
	clone := make(D3Slice, len(ds))
	for i, d := range ds {
		clone[i] = d.Clone()
	}
	return clone
}