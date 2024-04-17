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

func D3NumericalDifferentiation(x tensor.D3, f func(tensor.D3)float64) tensor.D3 {
	h := 0.001
	grad := tensor.NewD3ZerosLike(x)
	for i := range x {
		gradi := grad[i]
		xi := x[i]
		for j := range xi {
			gradij := gradi[j]
			xij := xi[j]
			for k := range xij {
				tmp := xij[k]

				xij[k] = tmp + h
				y1 := f(x)

				xij[k] = tmp - h
				y2 := f(x)

				gradij[k] = (y1 - y2) / (2*h)
				xij[k] = tmp
			}
		}
	}
	return grad
}