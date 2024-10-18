package mlfuncs3d

import (
	"github.com/sw965/crow/mlfuncs/2d"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw/fn"
	omwmath "github.com/sw965/omw/math"
)

func L2Regularization(c float64) func(tensor.D3) float64 {
	return func(w tensor.D3) float64 {
		return omwmath.Sum(fn.Map[tensor.D1](w, mlfuncs2d.L2Regularization(c))...)
	}
}

func L2RegularizationDerivative(c float64) func(tensor.D3) tensor.D3 {
	return func(w tensor.D3) tensor.D3 {
		return fn.Map[tensor.D3](w, mlfuncs2d.L2RegularizationDerivative(c))
	}
}

func NumericalDifferentiation(x tensor.D3, f func(tensor.D3) float64) tensor.D3 {
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

				gradij[k] = (y1 - y2) / (2 * h)
				xij[k] = tmp
			}
		}
	}
	return grad
}
