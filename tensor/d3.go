package tensor

import (
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"math/rand"
	crand "github.com/sw965/crow/math/rand"
)

type D3 struct {
	Channels      int
	Rows          int
	Cols          int
	ChannelStride int
	RowStride     int
	Data          []float32
}

func NewD3Zeros(chs, rows, cols int) D3 {
	rowStride := cols
	chStride := rows * rowStride
	n := chs * chStride
	return D3{
		Channels:      chs,
		Rows:          rows,
		Cols:          cols,
		ChannelStride: chStride,
		RowStride:     rowStride,
		Data:          make([]float32, n),
	}
}

func NewD3Ones(chs, rows, cols int) D3 {
	d3 := NewD3Zeros(chs, rows, cols)
	for i := range d3.Data {
		d3.Data[i] = 1.0
	}
	return d3
}

func NewD3Rademacher(chs, rows, cols int, rng *rand.Rand) D3 {
	d3 := NewD3Zeros(chs, rows, cols)
	for i := range d3.Data {
		d3.Data[i] = crand.Rademacher(rng)
	}
	return d3
}

func (d3 D3) NewZerosLike() D3 {
	return NewD3Zeros(d3.Channels, d3.Rows, d3.Cols)
}

func (d3 D3) NewOnesLike() D3 {
	return NewD3Ones(d3.Channels, d3.Rows, d3.Cols)
}

func (d3 D3) NewRademacherLike(rng *rand.Rand) D3 {
	return NewD3Rademacher(d3.Channels, d3.Rows, d3.Cols, rng)
}

func (d3 D3) N() int {
	return d3.Channels * d3.Rows * d3.Cols
}

func (d3 D3) Clone() D3 {
	return D3{
		Channels:      d3.Channels,
		Rows:          d3.Rows,
		Cols:          d3.Cols,
		ChannelStride: d3.ChannelStride,
		RowStride:     d3.RowStride,
		Data:          slices.Clone(d3.Data),
	}
}

func (d3 D3) At(ch, row, col int) int {
	return ch*d3.ChannelStride + row*d3.RowStride + col
}

func (d3 D3) ToD1() D1 {
	return D1{
		N:d3.N(),
		Inc:1,
		Data:slices.Clone(d3.Data),
	}
}

func (d3 D3) ToBlas32Vector() blas32.Vector {
	return blas32.Vector{
		N:    d3.N(),
		Inc:  1,
		Data: slices.Clone(d3.Data),
	}
}

func (d3 D3) Axpy(alpha float32, x D3) D3 {
	yv := d3.ToBlas32Vector()
	xv := x.ToBlas32Vector()
	blas32.Axpy(alpha, xv, yv)
	return D3{
		Channels:d3.Channels,
		Rows:d3.Rows,
		Cols:d3.Cols,
		ChannelStride:d3.ChannelStride,
		RowStride:d3.RowStride,
		Data:yv.Data,
	}
}

func (d3 D3) Scal(alpha float32) D3 {
	v := d3.ToBlas32Vector()
	blas32.Scal(alpha, v)
	return D3{
		Channels:d3.Channels,
		Rows:d3.Rows,
		Cols:d3.Cols,
		ChannelStride:d3.ChannelStride,
		RowStride:d3.RowStride,
		Data:v.Data,
	}
}

func (d3 D3) Transpose021() D3 {
    dst := NewD3Zeros(d3.Channels, d3.Cols, d3.Rows)
    dstChStride := dst.ChannelStride
    dstRowStride := dst.RowStride
    for ch := 0; ch < d3.Channels; ch++ {
        srcBase := ch * d3.ChannelStride
        dstBase := ch * dstChStride
        for col := 0; col < d3.Cols; col++ {
            srcOff := srcBase + col
            dstOff := dstBase + col*dstRowStride
            for row := 0; row < d3.Rows; row++ {
                dst.Data[dstOff+row] = d3.Data[srcOff+row*d3.RowStride]
            }
        }
    }
    return dst
}

func (d3 D3) Transpose102() D3 {
    dst := NewD3Zeros(d3.Rows, d3.Channels, d3.Cols)
    dstChStride := dst.ChannelStride
    dstRowStride := dst.RowStride
    for row := 0; row < d3.Rows; row++ {
        srcRowBase := row * d3.RowStride
        dstBase := row * dstChStride
        for ch := 0; ch < d3.Channels; ch++ {
            srcBase := srcRowBase + ch*d3.ChannelStride
            dstOff := dstBase + ch*dstRowStride
            copy(dst.Data[dstOff:dstOff+d3.Cols], d3.Data[srcBase:srcBase+d3.Cols])
        }
    }
    return dst
}

func (d3 D3) Transpose120() D3 {
    dst := NewD3Zeros(d3.Rows, d3.Cols, d3.Channels)
    dstChStride := dst.ChannelStride
    dstRowStride := dst.RowStride
    for row := 0; row < d3.Rows; row++ {
        srcRowBase := row * d3.RowStride
        dstBase := row * dstChStride
        for col := 0; col < d3.Cols; col++ {
            dstOff := dstBase + col*dstRowStride
            srcOff := srcRowBase + col
            for ch := 0; ch < d3.Channels; ch++ {
                dst.Data[dstOff+ch] = d3.Data[srcOff+ch*d3.ChannelStride]
            }
        }
    }
    return dst
}

func (d3 D3) Transpose201() D3 {
    dst := NewD3Zeros(d3.Cols, d3.Channels, d3.Rows)
    dstChStride := dst.ChannelStride
    dstRowStride := dst.RowStride
    for col := 0; col < d3.Cols; col++ {
        dstBase := col * dstChStride
        for ch := 0; ch < d3.Channels; ch++ {
            srcBase := ch*d3.ChannelStride + col
            dstOff := dstBase + ch*dstRowStride
            for row := 0; row < d3.Rows; row++ {
                dst.Data[dstOff+row] = d3.Data[srcBase+row*d3.RowStride]
            }
        }
    }
    return dst
}

func (d3 D3) Transpose210() D3 {
    dst := NewD3Zeros(d3.Cols, d3.Rows, d3.Channels)
    dstChStride := dst.ChannelStride
    dstRowStride := dst.RowStride
    for col := 0; col < d3.Cols; col++ {
        dstBase := col * dstChStride
        for row := 0; row < d3.Rows; row++ {
            dstOff := dstBase + row*dstRowStride
            srcBase := row*d3.RowStride + col
            for ch := 0; ch < d3.Channels; ch++ {
                dst.Data[dstOff+ch] = d3.Data[srcBase+ch*d3.ChannelStride]
            }
        }
    }
    return dst
}