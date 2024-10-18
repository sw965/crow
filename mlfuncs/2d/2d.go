package mlfuncs2d

import (
	"github.com/sw965/crow/mlfuncs/1d"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw/fn"
	omwmath "github.com/sw965/omw/math"
)

func SigmoidToTanh(y tensor.D2) tensor.D2 {
	return fn.Map[tensor.D2](y, mlfuncs1d.SigmoidToTanh)
}

func TanhToSigmoid(y tensor.D2) tensor.D2 {
	return fn.Map[tensor.D2](y, mlfuncs1d.TanhToSigmoid)
}

func L2Regularization(c float64) func(tensor.D2) float64 {
	return func(w tensor.D2) float64 {
		return omwmath.Sum(fn.Map[tensor.D1](w, mlfuncs1d.L2Regularization(c))...)
	}
}

func L2RegularizationDerivative(c float64) func(tensor.D2) tensor.D2 {
	return func(w tensor.D2) tensor.D2 {
		return fn.Map[tensor.D2](w, mlfuncs1d.L2RegularizationDerivative(c))
	}
}

func NumericalDifferentiation(x tensor.D2, f func(tensor.D2) float64) tensor.D2 {
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

			gradi[j] = (y1 - y2) / (2 * h)
			xi[j] = tmp
		}
	}
	return grad
}
