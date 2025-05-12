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

func (g General) Transpose(axes ...int) General {
	// ----- 1. 汎用設定 -----
	const ndim = 3
	if len(axes) == 0 {                 // 省略時は逆順
		axes = []int{2, 1, 0}
	}
	if len(axes) != ndim {
		panic("tensor3d: axes must have length 3")
	}

	// ----- 2. 軸番号の正規化と妥当性チェック -----
	seen := [ndim]bool{}
	for i, ax := range axes {
		if ax < 0 {
			ax += ndim
		}
		if ax < 0 || ax >= ndim || seen[ax] {
			panic("tensor3d: invalid axes permutation")
		}
		axes[i] = ax
		seen[ax] = true
	}

	// 元の次元サイズ
	srcShape := []int{g.Channels, g.Rows, g.Cols}

	// 出力テンソルの形状を決定
	dstCh, dstRows, dstCols := srcShape[axes[0]], srcShape[axes[1]], srcShape[axes[2]]
	dst := NewZeros(dstCh, dstRows, dstCols)

	// ----- 3. データをコピーしながら再配置 -----
	for ch := 0; ch < dstCh; ch++ {
		for r := 0; r < dstRows; r++ {
			for c := 0; c < dstCols; c++ {
				// dst インデックス (ch,r,c) を src インデックスにマップ
				srcIdx := [ndim]int{}
				srcIdx[axes[0]] = ch
				srcIdx[axes[1]] = r
				srcIdx[axes[2]] = c

				dst.Data[dst.At(ch, r, c)] = g.Data[g.At(srcIdx[0], srcIdx[1], srcIdx[2])]
			}
		}
	}

	return dst
}

// func (img *General) ToCol(filterRows, filterCols int) blas32.General {
// 	chs := img.Channels
// 	outRows := OutputRows(img, filterRows)
// 	outCols := OutputCols(img, filterCols)
// 	imgData := img.Data
// 	newData := make([]float32, outRows*outCols*chs*filterRows*filterCols)
// 	newIdx := 0

// 	for or := 0; or < outRows; or++ {
// 		for oc := 0; oc < outCols; oc++ {
// 			for ch := 0; ch < chs; ch++ {
// 				for fr := 0; fr < filterRows; fr++ {
// 					for fc := 0; fc < filterCols; fc++ {
// 						row := fr + or
// 						col := fc + oc
// 						imgIdx := img.FlatIndex(ch, row, col)
// 						newData[newIdx] = imgData[imgIdx]
// 						newIdx++
// 					}
// 				}
// 			}
// 		}
// 	}

// 	newCols := filterRows*filterCols*chs
// 	return blas32.General{
// 		Rows:outRows*outCols,
// 		Cols:newCols,
// 		Stride:newCols,
// 		Data:newData,
// 	}
// }