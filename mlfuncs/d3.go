package mlfuncs

import (
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func D3L2Regularization(w tensor.D3, l float64) float64 {
	l2 := make(tensor.D1, len(w))
	for i := range l2 {
		l2[i] = D2L2Regularization(w[i], l)
	}
	return omw.Sum(l2...)
}

func D3L2RegularizationDerivative(w tensor.D3, l float64) tensor.D3 {
	grad := make(tensor.D3, len(w))
	for i := range grad {
		grad[i] = D2L2RegularizationDerivative(w[i], l)
	}
	return grad
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