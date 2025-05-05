package vectors

import (
	"fmt"
	"gonum.org/v1/gonum/blas/blas32"
	"math/rand"
	"github.com/sw965/crow/blas32/vector"
)

func NewZerosLike(vs []blas32.Vector) []blas32.Vector {
	zeros := make([]blas32.Vector, len(vs))
	for i, v := range vs {
		zeros[i] = vector.NewZerosLike(v)
	}
	return zeros
}

func NewRademacherLike(vs []blas32.Vector, rng *rand.Rand) []blas32.Vector {
	rad := make([]blas32.Vector, len(vs))
	for i, v := range vs {
		rad[i] = vector.NewRademacherLike(v, rng)
	}
	return rad
}

func Clone(vs []blas32.Vector) []blas32.Vector {
	clone := make([]blas32.Vector, len(vs))
	for i, v := range vs {
		clone[i] = vector.Clone(v)
	}
	return clone
}

func Axpy(alpha float32, xs, ys []blas32.Vector) error {
	if len(xs) != len(ys) {
		return fmt.Errorf("vectors.Axpy len(xs) != len(ys)")
	}

	for i, x := range xs {
		y := ys[i]
		blas32.Axpy(alpha, x, y)
	}
	return nil
}

func Scal(alpha float32, ys []blas32.Vector) {
	for _, y := range ys {
		blas32.Scal(alpha, y)
	}
}