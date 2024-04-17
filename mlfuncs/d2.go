package mlfuncs

import (
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func D2L2Regularization(lambda float64) func(tensor.D2) float64 {
	return func(x tensor.D2) float64 {
		l2 := omw.MapFunc[tensor.D1](x, D1L2Regularization(lambda))
		return omw.Sum(l2...)
	}
}

func D2L2RegularizationDerivative(lambda float64) func(tensor.D2)tensor.D2 {
	return func(x tensor.D2) tensor.D2 {
		return omw.MapFunc[tensor.D2](x, D1L2RegularizationDerivative(lambda))
	}
}

func D2NumericalDifferentiation(x tensor.D2, f func(tensor.D2)float64) tensor.D2 {
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
	
			gradi[j] = (y1-y2) / (2*h)
			xi[j] = tmp
		}
	}
	return grad
}