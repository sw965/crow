package mlfuncs

import (
	"math"
)

func ScalarSigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func ScalarSigmoidGrad(y float64) float64 {
	return y * (1.0 - y)
}

func ScalarTanhGrad(y float64) float64 {
	return 1.0 - (y*y)
}

func ScalarTanhToSigmoid(y float64) float64 {
	return (y + 1.0) / 2.0
}

func ScalarNumericalDifferentiation(x float64, f func(float64)float64) float64 {
	h := 0.001
	y1 := f(x+h)
	y2 := f(x-h)
	return (y1-y2) / (2*h)
}