package scalar

import (
	"math"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidGrad(y float64) float64 {
	return y * (1.0 - y)
}

func SigmoidDerivative(x float64) float64 {
	y := Sigmoid(x)
	return SigmoidGrad(y)
}

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0
	}
}

func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}

func LeakyReLU(alpha float64) func(float64) float64 {
	return func(x float64) float64 {
		if x > 0 {
			return x
		} else {
			return x * alpha
		}
	}
}

func LeakyReLUDerivative(alpha float64) func(float64) float64 {
	return func(x float64) float64 {
		if x > 0 {
			return 1
		} else {
			return alpha
		}
	}
}

func NumericalDifferentiation(x float64, f func(float64) float64) float64 {
	h := 0.001
	y1 := f(x + h)
	y2 := f(x - h)
	return (y1 - y2) / (2 * h)
}
