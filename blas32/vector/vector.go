package vector

import (
	"fmt"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"slices"
	"math/rand"
	crand "github.com/sw965/crow/math/rand"
	"github.com/sw965/crow/blas32/tensor/2d"
	"github.com/sw965/crow/blas32/tensor/3d"
	"github.com/sw965/crow/blas32/tensor/4d"
	"github.com/chewxy/math32"
	omath "github.com/sw965/omw/math"
)

func NewZeros(n int) blas32.Vector {
	return blas32.Vector{
		N:    n,
		Inc:  1,
		Data: make([]float32, n),
	}
}

func NewZerosLike(vec blas32.Vector) blas32.Vector {
	return NewZeros(vec.N)
}

func NewRademacher(n int, rng *rand.Rand) blas32.Vector {
	vec := NewZeros(n)
	for i := range vec.Data {
		vec.Data[i] = crand.Rademacher(rng)
	}
	return vec
}

func NewRademacherLike(vec blas32.Vector, rng *rand.Rand) blas32.Vector {
	return NewRademacher(vec.N, rng)
}

func Clone(vec blas32.Vector) blas32.Vector {
	return blas32.Vector{
		N:vec.N,
		Inc:vec.Inc,
		Data:slices.Clone(vec.Data),
	}
}

func Reshape2D(vec blas32.Vector, rows, cols int) (blas32.General, error) {
	if vec.N != (rows*cols) {
		return blas32.General{}, fmt.Errorf("サイズが合わない")
	}

	return blas32.General{
		Rows:rows,
		Cols:cols,
		Stride:cols,
		Data:vec.Data,
	}, nil
}

func Reshape3D(vec blas32.Vector, chs, rows, cols int) (tensor3d.General, error) {
	n := chs*rows*cols
	if n != vec.N {
		return tensor3d.General{}, fmt.Errorf("サイズが合わない")
	}

	chStride := n / chs
	return tensor3d.General{
		Channels:chs,
		Rows:rows,
		Cols:cols,
		ChannelStride:chStride,
		RowStride:cols,
		Data:vec.Data,
	}, nil
}

func Reshaped4D(vec blas32.Vector, batches, chs, rows, cols int) (tensor4d.General, error) {
	n := batches * chs * rows * cols
	if n != vec.N  {
		return tensor4d.General{}, fmt.Errorf("サイズが合わない")
	}

	batchStride := n / batches
	chStride := batchStride / chs
	return tensor4d.General{
		Batches:batches,
		Channels:chs,
		Rows:rows,
		Cols:cols,
		BatchStride:batchStride,
		ChannelStride:chStride,
		RowStride:cols,
		Data:vec.Data,
	}, nil
}

func Hadamard(a, b blas32.Vector) (blas32.Vector, error) {
	if a.N != b.N {
		return blas32.Vector{}, fmt.Errorf("長さが一致しない")
	}
	newData := make([]float32, a.N)
	for i := range newData {
		newData[i] = a.Data[i] * b.Data[i]
	}
	return blas32.Vector{
		N:a.N,
		Inc:a.Inc,
		Data:newData,
	}, nil
}

func Standardize(x blas32.Vector) (blas32.Vector, error) {
	if x.N == 0 {
        return blas32.Vector{}, fmt.Errorf("vector length is zero")
    }

	mean := omath.Mean(x.Data...)

	//分散を求める
	meanDeviationSqSum := float32(0.0)
	for _, e := range x.Data {
		d := e - mean
		meanDeviationSqSum += d * d
	}
	vari := 1.0 / float32(x.N) * meanDeviationSqSum

	//標準偏差
	std := math32.Sqrt(vari + 1e-5)

	z := make([]float32, x.N)
	for i, e := range x.Data {
		z[i] = (e - mean) / std
	}

	return blas32.Vector{
		N:x.N,
		Inc:x.Inc,
		Data:z,
	}, nil
}

func StandardizeWithStats(x blas32.Vector) (blas32.Vector, float32, float32, error) {
	if x.N == 0 {
        return blas32.Vector{}, 0, 0, fmt.Errorf("vector length is zero")
    }

	mean := omath.Mean(x.Data...)

	//分散を求める
	meanDeviationSqSum := float32(0.0)
	for _, e := range x.Data {
		d := e - mean
		meanDeviationSqSum += d * d
	}
	vari := 1.0 / float32(x.N) * meanDeviationSqSum

	//標準偏差
	std := math32.Sqrt(vari + 1e-5)

	z := make([]float32, x.N)
	for i, e := range x.Data {
		z[i] = (e - mean) / std
	}

	return blas32.Vector{
		N:x.N,
		Inc:x.Inc,
		Data:z,
	}, mean, std, nil
}

func StandardizationDerivative(x blas32.Vector, mean, std float32) (blas32.General, error) {
    n := x.N
	if n == 0 {
        return blas32.General{}, fmt.Errorf("vector length is zero")
    }

    avg := 1.0 / float32(n)
    std3 := math32.Pow(std, 3)

    grad := blas32.General{
        Rows:  n,
        Cols:  n,
        Stride: n,
        Data:  make([]float32, n*n),
    }

    for i := 0; i < n; i++ {
        xid := x.Data[i] - mean
        for j := 0; j < n; j++ {
            var kd float32
            if i == j {
                kd = 1
            }
            xjd := x.Data[j] - mean
            one := (kd - avg) / std
            two := (xid * xjd) / (float32(n) * std3)
            idx := tensor2d.At(grad, i, j)
            grad.Data[idx] = one - two
        }
    }
    return grad, nil
}

func Outer(a, b blas32.Vector) blas32.General {
	y := blas32.General{
		Rows:   a.N,
		Cols:   b.N,
		Stride: b.N,
		Data:   make([]float32, a.N * b.N),
	}
	blas32.Ger(1.0, a, b, y)
	return y
}

func DotNoTrans2D(a blas32.Vector, b blas32.General) blas32.Vector {
	yn := b.Cols
	y := blas32.Vector{N: yn, Inc: 1, Data: make([]float32, yn)}
	blas32.Gemv(blas.Trans, 1.0, b, a, 1.0, y)
	return y
}

func DotTrans2D(a blas32.Vector, b blas32.General) blas32.Vector {
	yn := b.Rows
	y := blas32.Vector{N: yn, Inc: 1, Data: make([]float32, yn)}
	blas32.Gemv(blas.NoTrans, 1.0, b, a, 1.0, y)
	return y
}

func Softmax(x blas32.Vector) blas32.Vector {
    data := x.Data
    maxX := omath.Max(data...) // オーバーフロー対策
    expX := make([]float32, x.N)
    var sumExpX float32 = 0.0
    for i, e := range data {
        expX[i] = math32.Exp(e - maxX)
        sumExpX += expX[i]
    }

    y := make([]float32, x.N)
    for i := range expX {
        y[i] = expX[i] / sumExpX
    }

    return blas32.Vector{
		N:x.N,
		Inc:x.Inc,
		Data:y,
	}
}

func LeakyReLU(x blas32.Vector, alpha float32) blas32.Vector {
	y := make([]float32, x.N)
	for i, e := range x.Data {
		if e > 0 {
			y[i] = e
		} else {
			y[i] = alpha * e
		}
	}
	return blas32.Vector{
		N:x.N,
		Inc:x.Inc,
		Data:y,
	}
}

func LeakyReLUDerivative(x blas32.Vector, alpha float32) blas32.Vector {
	grad := make([]float32, x.N)
	for i, e := range x.Data {
		if e > 0 {
			grad[i] = 1.0
		} else {
			grad[i] = alpha
		}
	}
	return blas32.Vector{
		N:x.N,
		Inc:x.Inc,
		Data:grad,
	}
}

func CrossEntropy(y, t blas32.Vector) (float32, error) {
	ce := float32(0.0)
	e := float32(0.0001)
	for i := range y.Data {
		ye := omath.Max(y.Data[i], e)
		te := t.Data[i]
		ce += -te * math32.Log(ye)
	}
	return ce, nil
}

func SoftmaxCrossEntropyLossDerivative(y, t blas32.Vector) (blas32.Vector, error) {
	if y.N != t.N {
		return blas32.Vector{}, fmt.Errorf("要素数が一致しない")
	}
	grad := blas32.Vector{
		N:    y.N,
		Inc:  y.Inc,
		Data: make([]float32, y.N),
	}
	blas32.Copy(y, grad)
	blas32.Axpy(-1.0, t, grad)
	return grad, nil
}