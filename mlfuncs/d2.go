package mlfuncs

import (
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func D2L2Regularization(w tensor.D2, lambda float64) float64 {
	f := func(w tensor.D1) float64 { return D1L2Regularization(w, lambda) }
	l2 := omw.MapFunc[tensor.D1](w, f)
	return omw.Sum(l2...)
}

func D2L2RegularizationDerivative(w tensor.D2, lambda float64) tensor.D2 {
	f := func(w tensor.D1) tensor.D1 { return D1L2RegularizationDerivative(w, lambda) }
	return omw.MapFunc[tensor.D2](w, f)
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