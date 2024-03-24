package scalar

import (
	"math"
)

func TanhDerivative(x float64) float64 {
	y := math.Tanh(x)
	return TanhGrad(y)
}

func TanhGrad(y float64) float64 {
	return 1.0 - (y*y)
}

func TanhToSigmoidScale(y float64) float64 {
	return (y + 1.0) / 2.0
}

func LeakyReLU(alpha float64) func(float64)float64 {
	return func(x float64) float64 {
		if x >= 0 {
			return x
		} else {
			return x * alpha
		}
	}
}

func LeakyReLUDerivative(alpha float64) func(float64)float64 {
	return func(x float64) float64 {
		if x >= 0 {
			return 1.0
		} else {
			return alpha
		}
	}
}