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

func (ds D4Slice) Axpy(alpha float32, xs D4Slice) D4Slice {
	ys := make(D4Slice, len(ds))
	for i, y := range ds {
		ys[i] = y.Axpy(alpha, xs[i])
	}
	return ys
}

func (ds D4Slice) AxpyInPlace(alpha float32, xs D4Slice) {
	for i := range ds {
		ds[i].AxpyInPlace(alpha, xs[i])
	}
}

func (ds D4Slice) Scal(alpha float32) D4Slice {
	ys := make(D4Slice, len(ds))
	for i, y := range ds {
		ys[i] = y.Scal(alpha)
	}
	return ys
}

func (ds D4Slice) ScalInPlace(alpha float32) {
	for i := range ds {
		ds[i].ScalInPlace(alpha)
	}
}