package mlfuncs

import (
	"fmt"
	"math"

	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func D1Sigmoid(x tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](x, ScalarSigmoid)
}

func D1SigmoidGrad(y tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](y, ScalarSigmoidGrad)
}

func D1Tanh(x tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](x, math.Tanh)
}

func D1TanhGrad(y tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](y, ScalarTanhGrad)
}

func D1TanhToSigmoid(y tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](y, ScalarTanhToSigmoid)
}

func D1ReLU(x tensor.D1) tensor.D1 {
	y := make(tensor.D1, len(x))
	for i := range x {
		xi := x[i]
		if xi > 0 {
			y[i] = xi
		} else {
			y[i] = 0
		}
	}
	return y
}

func D1ReLUDerivative(x tensor.D1) tensor.D1 {
	grad := make(tensor.D1, len(x))
	for i := range x {
		if x[i] > 0 {
			grad[i] = 1
		} else {
			grad[i] = 0
		}
	}
	return grad
}

func D1LReLU(x tensor.D1, alpha float64) tensor.D1 {
	y := make(tensor.D1, len(x))
	for i := range x {
		xi := x[i]
		if xi > 0 {
			y[i] = xi
		} else {
			y[i] = xi * alpha
		}
	}
	return y
}

func D1LReLUDerivative(x tensor.D1, alpha float64) tensor.D1 {
	grad := make(tensor.D1, len(x))
	for i := range x {
		xi := x[i]
		if xi > 0 {
			grad[i] = 1
		} else {
			grad[i] = alpha
		}
	}
	return grad
}

func D1PReLU(x tensor.D1, alpha float64) tensor.D1 {
	return D1LReLU(x, alpha)
}

func D1PReLUDerivative(x tensor.D1, alpha float64) (tensor.D1, tensor.D1) {
	gradX := make(tensor.D1, len(x))
	vectorizedGradAlpha := make(tensor.D1, len(x))
	for i := range x {
		xi := x[i]
		if xi > 0 {
			gradX[i] = 1
			vectorizedGradAlpha[i] = 0
		} else {
			gradX[i] = alpha
			vectorizedGradAlpha[i] = xi
		}
	}
	return gradX, vectorizedGradAlpha
}

func D1MeanSquaredError(y, t tensor.D1) (float64, error) {
	if len(y) != len(t) {
		return 0.0, fmt.Errorf("len(y) != len(t) であるため、MeanSquaredErrorを計算できません。")
	}

    sum := 0.0
    for i := range y {
        diff := y[i] - t[i]
        sum += (diff * diff)
    }
    n := len(y)
    return 0.5 * sum / float64(n), nil
}

func D1MeanSquaredErrorDerivative(y, t tensor.D1) (tensor.D1, error) {
	if len(y) != len(t) {
		return tensor.D1{}, fmt.Errorf("len(y) != len(t) であるため、MeanSquaredErrorDerivativeを計算できません。")
	}

	n := len(y)
    grad := make(tensor.D1, n)
    for i := range y {
        grad[i] = (y[i] - t[i]) / float64(n)
    }
    return grad, nil
}

func D1L2Regularization(lambda float64) func(tensor.D1) float64 {
	return func(w tensor.D1) float64 {
		sum := 0.0
		for i := range w {
			wi := w[i]
			sum += wi * wi
		}
		return 0.5 * sum
	}
}

func D1L2RegularizationDerivative(lambda float64) func(tensor.D1) tensor.D1 {
	return func(w tensor.D1) tensor.D1 {
		grad := make(tensor.D1, len(w))
		for i := range grad {
			grad[i] = lambda * w[i]
		}
		return grad
	}
}

func D1NumericalDifferentiation(x tensor.D1, f func(tensor.D1)float64) tensor.D1 {
	h := 0.001
	grad := make(tensor.D1, len(x))
	for i := range x {
		tmp := x[i]

		x[i] = tmp + h
		y1 := f(x)

		x[i] = tmp - h
		y2 := f(x)

		grad[i] = (y1-y2) / (2*h)
		x[i] = tmp
	}
	return grad
}