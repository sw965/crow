package mlfuncs

import (
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func D3L2Regularization(lambda float64) func(tensor.D3) float64 {
	return func(w tensor.D3) float64 {
		l2 := omw.MapFunc[tensor.D1](w, D2L2Regularization(lambda))
		return omw.Sum(l2...)
	}
}

func D3L2RegularizationDerivative(lambda float64) func(tensor.D3) tensor.D3 {
	return func(w tensor.D3) tensor.D3 {
		return omw.MapFunc[tensor.D3](w, D2L2RegularizationDerivative(lambda))
	}
}