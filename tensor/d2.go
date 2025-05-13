package tensor

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"math"
	"math/rand"
	crand "github.com/sw965/crow/math/rand"
)

type D2 blas32.General

func NewD2Zeros(rows, cols int) D2 {
	return D2{
		Rows:   rows,
		Cols:   cols,
		Stride: cols,
		Data:   make([]float32, rows*cols),
	}
}

func NewD2Ones(rows, cols int) D2 {
	d2 := NewD2Zeros(rows, cols)
	for i := range d2.Data {
		d2.Data[i] = 1.0
	}
	return d2
}

func NewD2Rademacher(rows, cols int, rng *rand.Rand) D2 {
	d2 := NewD2Zeros(rows, cols)
	for i := range d2.Data {
		d2.Data[i] = crand.Rademacher(rng)
	}
	return d2
}

func NewD2He(rows, cols int, rng *rand.Rand) D2 {
    he := NewD2Zeros(rows, cols)
	fanIn := float64(rows)
    std := math.Sqrt(2.0 / fanIn)
    for i := range he.Data {
        he.Data[i] = float32(rng.NormFloat64() * std)
    }
    return he
}

func (d2 D2) NewZerosLike() D2 {
	return NewD2Zeros(d2.Rows, d2.Cols)
}

func (d2 D2) NewOnesLike() D2 {
	return NewD2Ones(d2.Rows, d2.Cols)
}

func (d2 D2) NewRademacherLike(rng *rand.Rand) D2 {
	return NewD2Rademacher(d2.Rows, d2.Cols, rng)
}

func (d2 D2) N() int {
	return d2.Rows * d2.Cols
}

func (d2 D2) Clone() D2 {
	return D2{
		Rows:   d2.Rows,
		Cols:   d2.Cols,
		Stride: d2.Stride,
		Data:   slices.Clone(d2.Data),
	}
}

func (d2 D2) At(row, col int) int {
	return row*d2.Stride + col
}

func (d2 D2) ToD1() D1 {
	return D1{
		N:d2.N(),
		Inc:1,
		Data:slices.Clone(d2.Data),
	}
}

func (d2 D2) ToBlas32Vector() blas32.Vector {
	v := blas32.Vector{
		N:d2.Rows*d2.Cols,
		Inc:1,
		Data:slices.Clone(d2.Data),
	}
	return v
}

func (d2 D2) Axpy(alpha float32, x D2) D2 {
	yv := d2.ToBlas32Vector()
	xv := x.ToBlas32Vector()
	blas32.Axpy(alpha, xv, yv)
	return D2{
		Rows:d2.Rows,
		Cols:d2.Cols,
		Stride:d2.Stride,
		Data:yv.Data,
	}
}

func (d2 D2) Scal(alpha float32) D2 {
	y := d2.ToBlas32Vector()
	blas32.Scal(alpha, y)
	return D2{
		Rows:d2.Rows,
		Cols:d2.Cols,
		Stride:d2.Stride,
		Data:y.Data,
	}
}

func (d2 D2) Sum0() D1 {
    sums := make([]float32, d2.Cols)
    for c := 0; c < d2.Cols; c++ {
        var sum float32
        for r := 0; r < d2.Rows; r++ {
            idx := d2.At(r, c)
            sum += d2.Data[idx]
        }
        sums[c] = sum
    }

    return D1{
        N:   d2.Cols,
        Inc: 1,
        Data: sums,
    }
}

func (d2 D2) Sum1() D1 {
	sums := make([]float32, d2.Rows)
	for r := 0; r < d2.Rows; r++ {
    	offset := r * d2.Stride
    	var sum float32
    	for c := 0; c < d2.Cols; c++ {
        	sum += d2.Data[offset+c]
    	}
    	sums[r] = sum
	}
	return D1{
		N:d2.Rows,
		Inc:1,
		Data:sums,
	}
}

func (d2 D2) Transpose() D2 {
	t := D2{
		Rows:d2.Cols,
		Cols:d2.Rows,
		Stride:d2.Rows,
		Data:make([]float32, d2.N()),
	}

	for i := range t.Rows {
		for j := range t.Cols {
			newIdx := t.At(i, j)
			oldIdx := d2.At(j, i)
			t.Data[newIdx] = d2.Data[oldIdx]
		}
	}
	return t
}

func (d2 D2) Dot(right D2, tL, tR blas.Transpose) D2 {
	y := blas32.General{
		Rows:d2.Rows,
		Cols:right.Cols,
		Stride:right.Cols,
		Data:make([]float32, d2.Rows*right.Cols),
	}
	blas32.Gemm(tL, tR, 1.0, blas32.General(d2), blas32.General(right), 0.0, y)
	return D2(y)
}