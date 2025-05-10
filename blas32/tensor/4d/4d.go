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

func (g General) Transpose(axes ...int) General {
	const ndim = 4

	// --- 1. デフォルトは軸を完全に逆順 ---
	if len(axes) == 0 {
		axes = []int{3, 2, 1, 0}
	}
	if len(axes) != ndim {
		panic("tensor4d: axes must have length 4")
	}

	// --- 2. 軸番号の正規化と妥当性チェック ---
	seen := [ndim]bool{}
	for i, ax := range axes {
		if ax < 0 {
			ax += ndim
		}
		if ax < 0 || ax >= ndim || seen[ax] {
			panic("tensor4d: invalid axes permutation")
		}
		axes[i] = ax
		seen[ax] = true
	}

	// --- 3. 出力テンソルの形状を決定 ---
	srcShape := []int{g.Batches, g.Channels, g.Rows, g.Cols}
	dstShape := [ndim]int{}
	for i := 0; i < ndim; i++ {
		dstShape[i] = srcShape[axes[i]]
	}
	dst := NewZeros(dstShape[0], dstShape[1], dstShape[2], dstShape[3])

	// --- 4. データをコピーしながら再配置 ---
	for b := 0; b < dstShape[0]; b++ {
		for c := 0; c < dstShape[1]; c++ {
			for r := 0; r < dstShape[2]; r++ {
				for col := 0; col < dstShape[3]; col++ {
					// dstインデックス(b,c,r,col) → srcインデックスへマッピング
					srcIdx := [ndim]int{}
					srcIdx[axes[0]] = b
					srcIdx[axes[1]] = c
					srcIdx[axes[2]] = r
					srcIdx[axes[3]] = col

					dst.Data[dst.At(b, c, r, col)] =
						g.Data[g.At(srcIdx[0], srcIdx[1], srcIdx[2], srcIdx[3])]
				}
			}
		}
	}

	return dst
}