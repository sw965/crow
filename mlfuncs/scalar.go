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

func ScalarSigmoidToTanhScale(y float64) float64 {
    return 2*y - 1.0
}

func ScalarTanhGrad(y float64) float64 {
	return 1.0 - (y*y)
}

func ScalarTanhToSigmoidScale(y float64) float64 {
	return (y + 1.0) / 2.0
}