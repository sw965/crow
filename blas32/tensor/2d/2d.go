package tensor2d

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"math"
	"math/rand"
	crand "github.com/sw965/crow/math/rand"
	"github.com/sw965/crow/blas32/tensor/3d"
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

func Sum0(gen blas32.General) blas32.Vector {
    sums := make([]float32, gen.Cols)
    for c := 0; c < gen.Cols; c++ {
        var sum float32
        for r := 0; r < gen.Rows; r++ {
            idx := At(gen, r, c)
            sum += gen.Data[idx]
        }
        sums[c] = sum
    }

    return blas32.Vector{
        N:   gen.Cols,
        Inc: 1,
        Data: sums,
    }
}

func Sum1(gen blas32.General) blas32.Vector {
	sums := make([]float32, gen.Rows)
	for r := 0; r < gen.Rows; r++ {
    	offset := r * gen.Stride
    	var sum float32
    	for c := 0; c < gen.Cols; c++ {
        	sum += gen.Data[offset+c]
    	}
    	sums[r] = sum
	}
	return blas32.Vector{
		N:gen.Rows,
		Inc:1,
		Data:sums,
	}
}

func Transpose(gen blas32.General) blas32.General {
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

func Dot(tA, tB blas.Transpose, a, b blas32.General) blas32.General {
	y := blas32.General{
		Rows:a.Rows,
		Cols:b.Cols,
		Stride:b.Cols,
		Data:make([]float32, a.Rows*b.Cols),
	}
	blas32.Gemm(tA, tB, 1.0, a, b, 0.0, y)
	return y
}

func Col2Im(col blas32.General, imgShape tensor3d.General, filterRows, filterCols int) tensor3d.General {
	chs := imgShape.Channels
	outRows := imgShape.ConvOutputRows(filterRows)
	outCols := imgShape.ConvOutputCols(filterCols)

	// （念のための簡易チェック）
	if col.Rows != outRows*outCols {
		panic("Col2Im: unexpected number of rows")
	}
	if col.Cols != chs*filterRows*filterCols {
		panic("Col2Im: unexpected number of cols")
	}

	recon := make([]float32, len(imgShape.Data))
	colIdx := 0

	for or := 0; or < outRows; or++ {
		for oc := 0; oc < outCols; oc++ {
			for ch := 0; ch < chs; ch++ {
				for fr := 0; fr < filterRows; fr++ {
					for fc := 0; fc < filterCols; fc++ {
						row := fr + or
						colPos := fc + oc
						imgIdx := imgShape.At(ch, row, colPos)
						recon[imgIdx] += col.Data[colIdx]
						colIdx++
					}
				}
			}
		}
	}

	return tensor3d.General{
		Channels:      imgShape.Channels,
		Rows:          imgShape.Rows,
		Cols:          imgShape.Cols,
		ChannelStride: imgShape.ChannelStride,
		RowStride:     imgShape.RowStride,
		Data:          recon,
	}
}