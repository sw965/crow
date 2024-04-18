package mlfuncs

import (
	"math"
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func D3L2Regularization(w tensor.D3, lambda float64) float64 {
	f := func(w tensor.D2) float64 { return D2L2Regularization(w, lambda) }
	l2 := omw.MapFunc[tensor.D1](w, f)
	return omw.Sum(l2...)
}

func D3L2RegularizationDerivative(w tensor.D3, lambda float64) tensor.D3 {
	f := func(w tensor.D2) tensor.D2 { return D2L2RegularizationDerivative(w, lambda) }
	return omw.MapFunc[tensor.D3](w, f)
}

func D3L2Norm(x tensor.D3) float64 {
	sum := 0.0
	for i := range x {
		xi := x[i]
		for j := range xi {
			xij := xi[j]
			for k := range xij {
				xijk := xij[k]
				sum += xijk * xijk
			}
		}
	}
	return math.Sqrt(sum)
}

func D3ClipL2Norm(x tensor.D3, threshold float64) tensor.D3 {
	norm := D3L2Norm(x)
	clipped := make(tensor.D3, len(x))
	for i := range x {
		clipped[i] = D2ClipL2Norm(x, threshold)
	}
	return clipped
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