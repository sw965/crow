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

func (ds D3Slice) Axpy(alpha float32, xs D3Slice) D3Slice {
	ys := make(D3Slice, len(ds))
	for i, d := range ds {
		ys[i] = d.Axpy(alpha, xs[i])
	}
	return ys
}

func (ds D3Slice) AxpyInPlace(alpha float32, xs D3Slice) {
	for i := range ds {
		ds[i].AxpyInPlace(alpha, xs[i])
	}
}

func (ds D3Slice) Scal(alpha float32) D3Slice {
	ys := make(D3Slice, len(ds))
	for i, d := range ds {
		ys[i] = d.Scal(alpha)
	}
	return ys
}

func (ds D3Slice) ScalInPlace(alpha float32) {
	for i := range ds {
		ds[i].ScalInPlace(alpha)
	}
}