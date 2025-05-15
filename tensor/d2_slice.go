package tensor

import (
	"math/rand"
)

type D2Slice []D2

func (ds D2Slice) NewZerosLike() D2Slice {
	zeros := make(D2Slice, len(ds))
	for i, d := range ds {
		zeros[i] = d.NewZerosLike()
	}
	return zeros
}

func (ds D2Slice) NewOnesLike() D2Slice {
	ones := make(D2Slice, len(ds))
	for i, d := range ds {
		ones[i] = d.NewOnesLike()
	}
	return ones
}

func (ds D2Slice) NewRademacherLike(rng *rand.Rand) D2Slice {
	rad := make(D2Slice, len(ds))
	for i, d := range ds {
		rad[i] = d.NewRademacherLike(rng)
	}
	return rad
}

func (ds D2Slice) Clone() D2Slice {
	clone := make(D2Slice, len(ds))
	for i, d := range ds {
		clone[i] = d.Clone()
	}
	return clone
}

func (ds D2Slice) Axpy(alpha float32, xs D2Slice) D2Slice {
	if len(ds) != len(xs) {
		panic("tensor2ds.Axpy len(xs) != len(ys)")
	}

	ys := make(D2Slice, len(ds))
	for i, d := range ds {
		ys[i] = d.Axpy(alpha, xs[i])
	}
	return ys
}

func (ds D2Slice) AxpyInPlace(alpha float32, xs D2Slice) {
	if len(ds) != len(xs) {
		panic("tensor2ds.Axpy len(xs) != len(ys)")
	}

	for i := range ds {
		ds[i].AxpyInPlace(alpha, xs[i])
	}
}

func (ds D2Slice) Scal(alpha float32) D2Slice {
	ys := make(D2Slice, len(ds))
	for i, d := range ds {
		ys[i] = d.Scal(alpha)
	}
	return ys
}

func (ds D2Slice) ScalInPlace(alpha float32) {
	for i := range ds {
		ds[i].ScalInPlace(alpha)
	}
}