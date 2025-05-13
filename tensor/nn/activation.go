package nn

import (
	"github.com/sw965/crow/tensor"
)

func ReLUD1WithAlpha(x tensor.D1, alpha float32) tensor.D1 {
	y := make([]float32, x.N)
	for i, e := range x.Data {
		if e > 0 {
			y[i] = e
		} else {
			y[i] = alpha * e
		}
	}

	x.Data = y
	return x
}

func LeakyReLUD1Derivative(x tensor.D1, alpha float32) tensor.D1 {
	grad := make([]float32, x.N)
	for i, e := range x.Data {
		if e > 0 {
			grad[i] = 1.0
		} else {
			grad[i] = alpha
		}
	}
	x.Data = grad
	return x
}