package tensors2d

import (
	"fmt"
	"gonum.org/v1/gonum/blas/blas32"
	"github.com/sw965/crow/blas32/tensor/2d"
	"math/rand"
)

func NewZerosLike(gens []blas32.General) []blas32.General {
	zeros := make([]blas32.General, len(gens))
	for i, gen := range gens {
		zeros[i] = tensor2d.NewZerosLike(gen)
	}
	return zeros
}

func NewOnesLike(gens []blas32.General) []blas32.General {
	ones := make([]blas32.General, len(gens))
	for i, gen := range gens {
		ones[i] = tensor2d.NewOnesLike(gen)
	}
	return ones
}

func NewRademacherLike(gens []blas32.General, rng *rand.Rand) []blas32.General {
	rad := make([]blas32.General, len(gens))
	for i, gen := range gens {
		rad[i] = tensor2d.NewRademacherLike(gen, rng)
	}
	return rad
}

func Clone(gens []blas32.General) []blas32.General {
	clone := make([]blas32.General, len(gens))
	for i, gen := range gens {
		clone[i] = tensor2d.Clone(gen)
	}
	return clone
}

func Axpy(alpha float32, xs, ys []blas32.General) error {
	if len(xs) != len(ys) {
		return fmt.Errorf("tensor2ds.Axpy len(xs) != len(ys)")
	}	
	for i, x := range xs {
		y := ys[i]
		tensor2d.Axpy(alpha, x, y)
	}
	return nil
}

func Scal(alpha float32, ys []blas32.General) {
	for _, y := range ys {
		tensor2d.Scal(alpha, y)
	}
}