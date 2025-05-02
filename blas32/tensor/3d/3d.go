package tensor3d

import (
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
)

type General struct {
	Channels      int
	Rows          int
	Cols          int
	ChannelStride int
	RowStride     int
	Data          []float32
}

func NewZeros(chs, rows, cols int) General {
	rowStride := cols
	chStride := rows * rowStride
	n := chs * chStride
	return General{
		Channels:      chs,
		Rows:          rows,
		Cols:          cols,
		ChannelStride: chStride,
		RowStride:     rowStride,
		Data:          make([]float32, n),
	}
}

func NewZerosLike(gen General) General {
	return NewZeros(gen.Channels, gen.Rows, gen.Cols)
}

func NewRademacher(chs, rows, cols int, rng *rand.Rand) General {
	gen := NewZeros(chs, rows, cols)
	for i := range gen.Data {
		gen.Data[i] = crand.Rademacher(rng)
	}
	return gen
}

func NewRademacherLike(gen General, rng *rand.Rand) General {
	return NewRademacher(gen.Channels, gen.Rows, gen.Cols, rng)
}

func (g General) N() int {
	return g.Channels * g.Rows * g.Cols
}

func (g General) Clone() General {
	return General{
		Channels:      g.Channels,
		Rows:          g.Rows,
		Cols:          g.Cols,
		ChannelStride: g.ChannelStride,
		RowStride:     g.RowStride,
		Data:          slices.Clone(g.Data),
	}
}

func (g General) At(ch, row, col int) int {
	return ch*g.ChannelStride + row*g.RowStride + col
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
