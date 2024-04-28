package mlfuncs

import (
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func D2SigmoidToTanh(y tensor.D2) tensor.D2 {
	return omw.MapFunc[tensor.D2](y, D1SigmoidToTanh)
}

func D2TanhToSigmoid(y tensor.D2) tensor.D2 {
	return omw.MapFunc[tensor.D2](y, D1TanhToSigmoid)
}

func D2LinearSum(x, w tensor.D2, b tensor.D1) (tensor.D1, error) {
	var err error
	y := make(tensor.D1, len(x))
	for i := range y {
		y[i], err = D1LinearSum(x[i], w[i], b[i])
		if err != nil {
			return tensor.D1{}, err
		}
	}
	return y, nil
}

func D2LinearSumDerivative(x, w tensor.D2) (tensor.D2, tensor.D2, tensor.D1, error) {
	n := len(x)
	gradX := make(tensor.D2, n)
	gradW := make(tensor.D2, n)
	gradB := make(tensor.D1, n)
	for i := range x {
		gradXi, gradWi, gradBi, err := D1LinearSumDerivative(x[i], w[i])
		if err != nil {
			return tensor.D2{}, tensor.D2{}, tensor.D1{}, err
		}
		gradX[i] = gradXi
		gradW[i] = gradWi
		gradB[i] = gradBi
	}
	return gradX, gradW, gradB, nil
}

func D2L2Regularization(w tensor.D2, l float64) float64 {
	l2 := make(tensor.D1, len(w))
	for i := range l2 {
		l2[i] = D1L2Regularization(w[i], l)
	}
	return omw.Sum(l2...)
}

func D2L2RegularizationDerivative(w tensor.D2, l float64) tensor.D2 {
	grad := make(tensor.D2, len(w))
	for i := range grad {
		grad[i] = D1L2RegularizationDerivative(w[i], l)
	}
	return grad
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