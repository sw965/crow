package vector

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"math/rand"
	crand "github.com/sw965/crow/math/rand"
)

func NewZeros(n int) blas32.Vector {
	return blas32.Vector{
		N:    n,
		Inc:  1,
		Data: make([]float32, n),
	}
}

func NewZerosLike(vec blas32.Vector) blas32.Vector {
	return NewZeros(vec.N)
}

func NewRademacher(n int, rng *rand.Rand) blas32.Vector {
	vec := NewZeros(n)
	for i := range vec.Data {
		vec.Data[i] = crand.Rademacher(rng)
	}
	return vec
}

func NewRademacherLike(vec blas32.Vector, rng *rand.Rand) blas32.Vector {
	return NewRademacher(vec.N, rng)
}

func Clone(vec blas32.Vector) blas32.Vector {
	return blas32.Vector{
		N:vec.N,
		Inc:vec.Inc,
		Data:slices.Clone(vec.Data),
	}
}

func Affine(x blas32.Vector, w blas32.General, b blas32.Vector) blas32.Vector {
	yn := len(b.Data)
	y := blas32.Vector{N: yn, Inc: 1, Data: make([]float32, yn)}
	blas32.Copy(b, y)
	blas32.Gemv(blas.Trans, 1.0, w, x, 1.0, y)
	return y
}
