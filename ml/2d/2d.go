package ml2d

import (
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/tensor"
)

func LinearSum(x, w tensor.D2, b tensor.D1) (tensor.D1, error) {
	y := make(tensor.D1, len(x))
	var err error
	for i, xi := range x {
		wi := w[i]
		bi := b[i]
		y[i], err = ml1d.LinearSum(xi, wi, bi)
		if err != nil {
			return nil, err
		}
	}
	return y, nil
}

//バイアス項(b)の微分は常に1であり、連鎖律において計算する必要がない為、計算を省く。
func LinearSumDerivative(x, w tensor.D2) (tensor.D2, tensor.D2, error) {
	n := len(x)
	gradX := make(tensor.D2, n)
	gradW := make(tensor.D2, n)

	for i := 0; i < n; i++ {
		gradXi, gradWi, err := ml1d.LinearSumDerivative(x[i], w[i])
		if err != nil {
			return nil, nil, err
		}

		gradX[i] = gradXi
		gradW[i] = gradWi
	}
	return gradX, gradW, nil
}

func NumericalDifferentiation(x tensor.D2, f func(tensor.D2) float64) tensor.D2 {
	h := 0.001
	grad := tensor.NewD2ZerosLike(x)
	for i := range x {
		gradi := grad[i]
		xi := x[i]
		for j := range xi {
			tmp := xi[j]

			xi[j] = tmp + h
			y1 := f(x)

			xi[j] = tmp - h
			y2 := f(x)

			gradi[j] = (y1 - y2) / (2 * h)
			xi[j] = tmp
		}
	}
	return grad
}
