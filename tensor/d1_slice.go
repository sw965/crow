package tensor

import (
	"math/rand"
)

type D1Slice []D1

func (ds D1Slice) NewZerosLike() D1Slice {
	zeros := make(D1Slice, len(ds))
	for i, d := range ds {
		zeros[i] = d.NewZerosLike()
	}
	return zeros
}

func (ds D1Slice) NewOnesLike() D1Slice {
	ones := make(D1Slice, len(ds))
	for i, d := range ds {
		ones[i] = d.NewOnesLike()
	}
	return ones
}

func (ds D1Slice) NewRademacherLike(rng *rand.Rand) D1Slice {
	rad := make(D1Slice, len(ds))
	for i, d := range ds {
		rad[i] = d.NewRademacherLike(rng)
	}
	return rad
}

func (ds D1Slice) Clone() D1Slice {
	clone := make(D1Slice, len(ds))
	for i, d := range ds {
		clone[i] = d.Clone()
	}
	return clone
}

func (ds D1Slice) Axpy(alpha float32, xs D1Slice) D1Slice {
	if len(ds) != len(xs) {
		panic("vectors.Axpy len(xs) != len(ys)")
	}

	ys := make(D1Slice, len(ds))
	for i, d := range ds {
		ys[i] = d.Axpy(alpha, xs[i])
	}
	return ys
}

func (ds D1Slice) AxpyInPlace(alpha float32, xs D1Slice) {
	if len(ds) != len(xs) {
		panic("vectors.Axpy len(xs) != len(ys)")
	}
	for i := range ds {
		ds[i].AxpyInPlace(alpha, xs[i])
	}
}

func (ds D1Slice) Scal(alpha float32) D1Slice {
	ys := make(D1Slice, len(ds))
	for i, d := range ds {
		ys[i] = d.Scal(alpha)
	}
	return ys
}

func (ds D1Slice) ScalInPlace(alpha float32) {
	for i := range ds {
		ds[i].ScalInPlace(alpha)
	}
}