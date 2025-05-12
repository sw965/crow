package tensor3d

import (
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"math/rand"
	crand "github.com/sw965/crow/math/rand"
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

func NewOnes(chs, rows, cols int) General {
	gen := NewZeros(chs, rows, cols)
	for i := range gen.Data {
		gen.Data[i] = 1.0
	}
	return gen
}

func NewOnesLike(gen General) General {
	return NewOnes(gen.Channels, gen.Rows, gen.Cols)
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

func (g General) Flatten() blas32.Vector {
	return blas32.Vector{
		N:    g.N(),
		Inc:  1,
		Data: slices.Clone(g.Data),
	}
}

func (g General) Axpy(alpha float32, x General) {
	xv := x.ToVector()
	yv := g.ToVector()
	blas32.Axpy(alpha, xv, yv)
}

func (g *General) Transpose021() General {
    dst := NewZeros(g.Channels, g.Cols, g.Rows)
    dstChStride := dst.ChannelStride
    dstRowStride := dst.RowStride
    for ch := 0; ch < g.Channels; ch++ {
        srcBase := ch * g.ChannelStride
        dstBase := ch * dstChStride
        for col := 0; col < g.Cols; col++ {
            srcOff := srcBase + col
            dstOff := dstBase + col*dstRowStride
            for row := 0; row < g.Rows; row++ {
                dst.Data[dstOff+row] = g.Data[srcOff+row*g.RowStride]
            }
        }
    }
    return dst
}

func (g *General) Transpose102() General {
    dst := NewZeros(g.Rows, g.Channels, g.Cols)
    dstChStride := dst.ChannelStride
    dstRowStride := dst.RowStride
    for row := 0; row < g.Rows; row++ {
        srcRowBase := row * g.RowStride
        dstBase := row * dstChStride
        for ch := 0; ch < g.Channels; ch++ {
            srcBase := srcRowBase + ch*g.ChannelStride
            dstOff := dstBase + ch*dstRowStride
            copy(dst.Data[dstOff:dstOff+g.Cols], g.Data[srcBase:srcBase+g.Cols])
        }
    }
    return dst
}

func (g *General) Transpose120() General {
    dst := NewZeros(g.Rows, g.Cols, g.Channels)
    dstChStride := dst.ChannelStride
    dstRowStride := dst.RowStride
    for row := 0; row < g.Rows; row++ {
        srcRowBase := row * g.RowStride
        dstBase := row * dstChStride
        for col := 0; col < g.Cols; col++ {
            dstOff := dstBase + col*dstRowStride
            srcOff := srcRowBase + col
            for ch := 0; ch < g.Channels; ch++ {
                dst.Data[dstOff+ch] = g.Data[srcOff+ch*g.ChannelStride]
            }
        }
    }
    return dst
}

func (g *General) Transpose201() General {
    dst := NewZeros(g.Cols, g.Channels, g.Rows)
    dstChStride := dst.ChannelStride
    dstRowStride := dst.RowStride
    for col := 0; col < g.Cols; col++ {
        dstBase := col * dstChStride
        for ch := 0; ch < g.Channels; ch++ {
            srcBase := ch*g.ChannelStride + col
            dstOff := dstBase + ch*dstRowStride
            for row := 0; row < g.Rows; row++ {
                dst.Data[dstOff+row] = g.Data[srcBase+row*g.RowStride]
            }
        }
    }
    return dst
}

func (g *General) Transpose210() General {
    dst := NewZeros(g.Cols, g.Rows, g.Channels)
    dstChStride := dst.ChannelStride
    dstRowStride := dst.RowStride
    for col := 0; col < g.Cols; col++ {
        dstBase := col * dstChStride
        for row := 0; row < g.Rows; row++ {
            dstOff := dstBase + row*dstRowStride
            srcBase := row*g.RowStride + col
            for ch := 0; ch < g.Channels; ch++ {
                dst.Data[dstOff+ch] = g.Data[srcBase+ch*g.ChannelStride]
            }
        }
    }
    return dst
}

//cnn用のメソッド。レシーバーの名前はimgとする。

func (img *General) ZeroPadding2D(top, bot, left, right int) General {
    padded := NewZeros(img.Channels, img.Rows+top+bot, img.Cols+left+right)
    for ch := 0; ch < img.Channels; ch++ {
        for row := 0; row < img.Rows; row++ {
            for col := 0; col < img.Cols; col++ {
                oldIdx := img.At(ch, row, col)
                newIdx := padded.At(ch, row+top, col+left)
                padded.Data[newIdx] = img.Data[oldIdx]
            }
        }
    }
    return padded
}

func (img *General) SameZeroPadding2D(filterRows, filterCols int) General {
	top := (filterRows - 1) / 2
    bot := filterRows - 1 - top
    left := (filterCols - 1) / 2
    right := filterCols - 1 - left
    return img.ZeroPadding2D(top, bot, left, right)
}

func (img *General) ConvOutputRows(filterRows int) int {
	return img.Rows - filterRows + 1
}

func (img *General) ConvOutputCols(filterCols int) int {
	return img.Cols - filterCols + 1
}

func (img *General) ToCol(filterRows, filterCols int) blas32.General {
	chs := img.Channels
	outRows := img.ConvOutputRows(filterRows)
	outCols := img.ConvOutputCols(filterCols)
	imgData := img.Data
	newData := make([]float32, outRows*outCols*chs*filterRows*filterCols)
	newIdx := 0

	for or := 0; or < outRows; or++ {
		for oc := 0; oc < outCols; oc++ {
			for ch := 0; ch < chs; ch++ {
				for fr := 0; fr < filterRows; fr++ {
					for fc := 0; fc < filterCols; fc++ {
						row := fr + or
						col := fc + oc
						imgIdx := img.At(ch, row, col)
						newData[newIdx] = imgData[imgIdx]
						newIdx++
					}
				}
			}
		}
	}

	newCols := filterRows*filterCols*chs
	return blas32.General{
		Rows:outRows*outCols,
		Cols:newCols,
		Stride:newCols,
		Data:newData,
	}
}