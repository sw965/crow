package mlfuncs

import (
	"github.com/sw965/crow/tensor"
)

func D2L2Regularization(w tensor.D2, lambda float64) float64 {
	sum := 0.0
	for i := range w {
		sum += D1L2Regularization(w[i], lambda)
	}
	return lambda * sum
}

func D2L2RegularizationDerivative(w tensor.D2, lambda float64) tensor.D2 {
	grad := make(tensor.D2, len(w))
	for i := range w {
		grad[i] = D1L2RegularizationDerivative(w[i], lambda)
	}
	return grad
}