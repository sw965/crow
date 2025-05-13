package math

import (
	"github.com/sw965/crow/tensor"
)

func Standardize(x tensor.D1) tensor.D1 {
	if x.N == 0 {
        panic("vector length is zero")
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

	x.Data = z
	return x
}

func StandardizeWithStats(x tensor.D1) (D1, float32, float32) {
	if x.N == 0 {
        panic("vector length is zero")
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

	x.Data = z
	return x, mean, std
}

func StandardizationDerivative(x tensor.D1, mean, std float32) tensor.D1 {
	n := x.N
	if n == 0 {
        panic("vector length is zero")
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

func Softmax(x tensor.D1) tensor.D1 {
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

	x.Data = y
	return x
}

func CrossEntropy(y, t tensor.D1) float32 {
	ce := float32(0.0)
	for i := range d1.Data {
		ye := omath.Max(d1.Data[i], 0.0001)
		te := t.Data[i]
		ce += -te * math32.Log(ye)
	}
	return ce, nil
}

func SoftmaxCrossEntropyLossDerivative(y, t tensor.D1) tensor.D1 {
	if y.N != t.N {
		panic("要素数が一致しない")
	}
	grad := blas32.Vector{
		N:    y.N,
		Inc:  y.Inc,
		Data: make([]float32, y.N),
	}
	blas32.Copy(blas32.Vector(y), grad)
	blas32.Axpy(-1.0, blas32.Vector(t), grad)
	return grad
}