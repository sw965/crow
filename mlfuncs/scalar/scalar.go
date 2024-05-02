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

func SigmoidToTanh(y float64) float64 {
	return 2*y - 1.0
}

func TanhGrad(y float64) float64 {
	return 1.0 - (y*y)
}

func TanhDerivative(x float64) float64 {
	y := math.Tanh(x)
	return TanhGrad(y)
}

func TanhToSigmoid(y float64) float64 {
	return (y + 1.0) / 2.0
}

func NumericalDifferentiation(x float64, f func(float64)float64) float64 {
	h := 0.001
	y1 := f(x+h)
	y2 := f(x-h)
	return (y1-y2) / (2*h)
}