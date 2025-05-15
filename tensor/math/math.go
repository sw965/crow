package math

import (
	"github.com/sw965/crow/tensor"
	omath "github.com/sw965/omw/math"
	"github.com/chewxy/math32"
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

func StandardizeWithStats(x tensor.D1) (tensor.D1, float32, float32) {
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

func StandardizationDerivative(x tensor.D1, mean, std float32) tensor.D2 {
	n := x.N
	if n == 0 {
        panic("vector length is zero")
    }

    avg := 1.0 / float32(n)
    std3 := math32.Pow(std, 3)

    grad := tensor.D2{
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
            idx := grad.At(i, j)
            grad.Data[idx] = one - two
        }
    }
    return grad
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
	for i := range y.Data {
		ye := omath.Max(y.Data[i], 0.0001)
		te := t.Data[i]
		ce += -te * math32.Log(ye)
	}
	return ce
}

func SoftmaxCrossEntropyLossDerivative(y, t tensor.D1) tensor.D1 {
	if y.N != t.N {
		panic("要素数が一致しない")
	}
	//y - t
	return y.Axpy(-1.0, t)
}

func SumSquaredLoss(y, t tensor.D1) float32 {
	if y.N != t.N {
		panic("len(y) != len(t) であるため、SumSquaredErrorを計算できません。")
	}
	var sqSum float32 = 0.0
	for i := range y.Data {
		diff := y.Data[i] - t.Data[i]
		sqSum += (diff * diff)
	}
	return 0.5 * sqSum
}

func SumSquaredLossDerivative(y, t tensor.D1) tensor.D1 {
	return y.Axpy(-1.0, t)
}