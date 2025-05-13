package tensor

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"math/rand"
	crand "github.com/sw965/crow/math/rand"
	oslices "github.com/sw965/omw/slices"
)

type D1 blas32.Vector

func NewD1Zeros(n int) D1 {
	return D1{
		N:    n,
		Inc:  1,
		Data: make([]float32, n),
	}
}

func NewD1Ones(n int) D1 {
	d1 := NewD1Zeros(n)
	for i := range d1.Data {
		d1.Data[i] = 1.0
	}
	return d1
}

func NewD1Rademacher(n int, rng *rand.Rand) D1 {
	d1 := NewD1Zeros(n)
	for i := range d1.Data {
		d1.Data[i] = crand.Rademacher(rng)
	}
	return d1
}


func (d1 D1) NewZerosLike() D1 {
	return NewD1Zeros(d1.N)
}

func (d1 D1) NewOnesLike() D1 {
	return NewD1Ones(d1.N)
}

func (d1 D1) NewRademacherLike(rng *rand.Rand) D1 {
	return NewD1Rademacher(d1.N, rng)
}

func (d1 D1) Clone() D1 {
	return D1{
		N:d1.N,
		Inc:d1.Inc,
		Data:slices.Clone(d1.Data),
	}
}

func (d1 D1) Axpy(alpha float32, x D1) D1 {
	d1.Data = slices.Clone(d1.Data)
	blas32.Axpy(alpha, blas32.Vector(x), blas32.Vector(d1))
	return d1
}

func (d1 D1) Scal(alpha float32) D1 {
	d1.Data = slices.Clone(d1.Data)
	blas32.Scal(alpha, blas32.Vector(d1))
	return d1
}

func (d1 D1) Hadamard(x D1) D1 {
	if d1.N != x.N {
		panic("長さが一致しないのでアダマール積が出来ない")
	}

	newData := make([]float32, d1.N)
	for i := range newData {
		newData[i] = d1.Data[i] * x.Data[i]
	}
	d1.Data = newData
	return d1
}

func (d1 D1) Reshape2D(rows, cols int) D2 {
	if rows == -1 && cols == -1 {
		panic("自動サイズ指定は1つまで")
	}

	if rows == -1 {
		rows = d1.N / cols
	}

	if cols == -1 {
		cols = d1.N / rows
	}

	if d1.N != (rows*cols) {
		panic("サイズが合わない")
	}

	return D2{
		Rows:rows,
		Cols:cols,
		Stride:cols,
		Data:slices.Clone(d1.Data),
	}
}

func (d1 D1) Reshape3D(chs, rows, cols int) D3 {
	if oslices.Count([]int{chs, rows, cols}, -1) > 1 {
		panic("自動サイズの指定は1つまで")
	}

	if chs == -1 {
		chs = d1.N / (rows * cols)
	}

	if rows == -1 {
		rows = d1.N / (chs * cols)
	}

	if cols == -1 {
		cols = d1.N / (chs * rows)
	}

	n := chs*rows*cols
	if n != d1.N {
		panic("サイズが合わない")
	}

	chStride := n / chs
	return D3{
		Channels:chs,
		Rows:rows,
		Cols:cols,
		ChannelStride:chStride,
		RowStride:cols,
		Data:slices.Clone(d1.Data),
	}
}

func (d1 D1) Reshape4D(batches, chs, rows, cols int) D4 {
	if oslices.Count([]int{batches, chs, rows, cols}, -1) > 1 {
		panic("自動サイズの指定は1つまで")
	}

	if batches == -1 {
		batches = d1.N / (chs * rows * cols)
	}

	if chs == -1 {
		chs = d1.N / (batches * rows * cols)
	}

	if rows == -1 {
		rows = d1.N / (batches * chs * cols)
	}

	if cols == -1 {
		cols = d1.N / (batches * chs * rows)
	}

	n := batches * chs * rows * cols
	if n != d1.N  {
		panic("サイズが合わない")
	}

	batchStride := n / batches
	chStride := batchStride / chs
	return D4{
		Batches:batches,
		Channels:chs,
		Rows:rows,
		Cols:cols,
		BatchStride:batchStride,
		ChannelStride:chStride,
		RowStride:cols,
		Data:slices.Clone(d1.Data),
	}
}

func (d1 D1) Outer(right D1) D2 {
	y := blas32.General{
		Rows:   d1.N,
		Cols:   right.N,
		Stride: right.N,
		Data:   make([]float32, d1.N * right.N),
	}
	blas32.Ger(1.0, blas32.Vector(d1), blas32.Vector(right), y)
	return D2(y)
}

func (d1 D1) DotNoTrans2D(d2 D2) D1 {
	yn := d2.Cols
	y := blas32.Vector{N: yn, Inc: 1, Data: make([]float32, yn)}
	blas32.Gemv(blas.Trans, 1.0, blas32.General(d2), blas32.Vector(d1), 0.0, y)
	return D1(y)
}

func (d1 D1) DotTrans2D(d2 D2) D1 {
	yn := d2.Rows
	y := blas32.Vector{N: yn, Inc: 1, Data: make([]float32, yn)}
	blas32.Gemv(blas.NoTrans, 1.0, blas32.General(d2), blas32.Vector(d1), 0.0, y)
	return D1(y)
}