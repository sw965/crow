package tensor2d

import (
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"math"
	"math/rand"
	crand "github.com/sw965/crow/math/rand"
)

func NewZeros(rows, cols int) blas32.General {
	return blas32.General{
		Rows:   rows,
		Cols:   cols,
		Stride: cols,
		Data:   make([]float32, rows*cols),
	}
}

func NewZerosLike(gen blas32.General) blas32.General {
	return NewZeros(gen.Rows, gen.Cols)
}

func NewOnes(rows, cols int) blas32.General {
	gen := NewZeros(rows, cols)
	for i := range gen.Data {
		gen.Data[i] = 1.0
	}
	return gen
}

func NewOnesLike(gen blas32.General) blas32.General {
	return NewOnes(gen.Rows, gen.Cols)
}

func NewHe(rows, cols int, rng *rand.Rand) blas32.General {
    gen := NewZeros(rows, cols)
	fanIn := float64(rows)
    std := math.Sqrt(2.0 / fanIn)
    for i := range gen.Data {
        gen.Data[i] = float32(rng.NormFloat64() * std)
    }
    return gen
}

func NewRademacher(rows, cols int, rng *rand.Rand) blas32.General {
	gen := NewZeros(rows, cols)
	for i := range gen.Data {
		gen.Data[i] = crand.Rademacher(rng)
	}
	return gen
}

func NewRademacherLike(gen blas32.General, rng *rand.Rand) blas32.General {
	return NewRademacher(gen.Rows, gen.Cols, rng)
}

func N(gen blas32.General) int {
	return gen.Rows * gen.Cols
}

func Clone(gen blas32.General) blas32.General {
	return blas32.General{
		Rows:   gen.Rows,
		Cols:   gen.Cols,
		Stride: gen.Stride,
		Data:   slices.Clone(gen.Data),
	}
}

func At(gen blas32.General, row, col int) int {
	return row*gen.Stride + col
}

func ToVector(gen blas32.General) blas32.Vector {
	return blas32.Vector{
		N:    N(gen),
		Inc:  1,
		Data: gen.Data,
	}
}

func Flatten(gen blas32.General) blas32.Vector {
	return blas32.Vector{
		N:N(gen),
		Inc:1,
		Data:slices.Clone(gen.Data),
	}
}

func Scal(alpha float32, gen blas32.General) {
	vec := ToVector(gen)
	blas32.Scal(alpha, vec)
}

func Axpy(alpha float32, x, y blas32.General) {
	xv := ToVector(x)
	yv := ToVector(y)
	blas32.Axpy(alpha, xv, yv)
}

func T(gen blas32.General) blas32.General {
	t := blas32.General{
		Rows:gen.Cols,
		Cols:gen.Rows,
		Stride:gen.Rows,
		Data:make([]float32, N(gen)),
	}

	for i := range t.Rows {
		for j := range t.Cols {
			newIdx := At(t, i, j)
			oldIdx := At(gen, j, i)
			t.Data[newIdx] = gen.Data[oldIdx]
		}
	}
	return t
}