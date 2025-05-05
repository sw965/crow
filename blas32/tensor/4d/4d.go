package tensor4d

import (
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"math"
	"math/rand"
	crand "github.com/sw965/crow/math/rand"
)

type General struct {
	Batches       int
	Channels      int
	Rows          int
	Cols          int
	BatchStride   int
	ChannelStride int
	RowStride     int
	Data          []float32
}

func NewZeros(batches, chs, rows, cols int) General {
	rowStride := cols
	chStride := rows * rowStride
	batchStride := chs * chStride
	n := batches * batchStride

	return General{
		Batches:       batches,
		Channels:      chs,
		Rows:          rows,
		Cols:          cols,
		BatchStride:   batchStride,
		ChannelStride: chStride,
		RowStride:     rowStride,
		Data:          make([]float32, n),
	}
}

func NewZerosLike(gen General) General {
	return NewZeros(gen.Batches, gen.Channels, gen.Rows, gen.Cols)
}

func NewHe(batches, chs, rows, cols int, rng *rand.Rand) General {
    gen := NewZeros(batches, chs, rows, cols)
    fanIn := float64(chs * rows * cols)
    std := float32(math.Sqrt(2.0 / fanIn))
    for i := range gen.Data {
        gen.Data[i] = float32(rng.NormFloat64()) * std
    }
    return gen
}

func NewRademacher(batches, chs, rows, cols int, rng *rand.Rand) General {
	gen := NewZeros(batches, chs, rows, cols)
	for i := range gen.Data {
		gen.Data[i] = crand.Rademacher(rng)
	}
	return gen
}

func NewRademacherLike(gen General, rng *rand.Rand) General {
	return NewRademacher(gen.Batches, gen.Channels, gen.Rows, gen.Cols, rng)
}

func (g General) N() int {
	return g.Batches * g.Channels * g.Rows * g.Cols
}

func (g General) Clone() General {
	return General{
		Batches:       g.Batches,
		Channels:      g.Channels,
		Rows:          g.Rows,
		Cols:          g.Cols,
		BatchStride:   g.BatchStride,
		ChannelStride: g.ChannelStride,
		RowStride:     g.RowStride,
		Data:          slices.Clone(g.Data),
	}
}

func (g General) At(batch, ch, row, col int) int {
	return (batch * g.BatchStride) + (ch * g.ChannelStride) + (row * g.RowStride) + col
}

func (g General) ToVector() blas32.Vector {
	return blas32.Vector{
		N:    g.N(),
		Inc:  1,
		Data: g.Data,
	}
}

func (g General) Axpy(alpha float32, x General) {
	xv := x.ToVector()
	yv := g.ToVector()
	blas32.Axpy(alpha, xv, yv)
}
