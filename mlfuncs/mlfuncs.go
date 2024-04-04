package mlfuncs

import (
	"fmt"
	"math"

	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func ScalarSigmoidToTanhScale(y float64) float64 {
    return 2*y - 1.0
}

func ScalarTanhGrad(y float64) float64 {
	return 1.0 - (y*y)
}

func ScalarTanhToSigmoidScale(y float64) float64 {
	return (y + 1.0) / 2.0
}

func D1SigmoidToTanhScale(y tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](y, ScalarSigmoidToTanhScale)
}

func D1Tanh(x tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](x, math.Tanh)
}

func D1TanhGrad(y tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](y, ScalarTanhGrad)
}

func D1TanhToSigmoidScale(y tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](y, ScalarTanhToSigmoidScale)
}

func D1PRReLU(x, alpha, beta tensor.D1, gamma float64, isTraining bool) (tensor.D1, error) {
	if len(x) != len(alpha) {
		return tensor.D1{}, fmt.Errorf("xとalphaの長さが異なるため、D1PRReLUを計算できません。")
	}

	if len(x) != len(beta) {
		return tensor.D1{}, fmt.Errorf("xとbetaの長さが異なるため、D1PRReLUを計算できません。")
	}

	y := make(tensor.D1, len(x))
	if isTraining {
		for i := range y {
			xi := x[i]
			if xi >= 0 {
				y[i] = xi
			} else {
				y[i] = xi * alpha[i] * beta[i]
			}
		}
	} else {
		for i := range y {
			xi := x[i]
			if xi >= 0 {
				y[i] = xi
			} else {
				y[i] = xi * alpha[i] * gamma
			}
		}
	}
	return y, nil
}

func D1MeanSquaredError(y, t tensor.D1) float64 {
    sum := 0.0
    for i := range y {
        diff := y[i] - t[i]
        sum += (diff * diff)
    }
    n := len(y)
    return 0.5 * sum / float64(n)
}

func D1MeanSquaredErrorDerivative(y, t tensor.D1) tensor.D1 {
	n := len(y)
    grad := make(tensor.D1, n)
    for i := range y {
        grad[i] = (y[i] - t[i]) / float64(n)
    }
    return grad
}

func D1L2Regularization(w tensor.D1, lambda float64) float64 {
	sum := 0.0
	for _, ele := range w {
		sum += ele * ele
	}
	return 0.5 * lambda * sum
}

func D1L2RegularizationDerivative(w tensor.D1, lambda float64) tensor.D1 {
	grad := make(tensor.D1, len(w))
	for i := range w {
		grad[i] = lambda * w[i]
	}
	return grad
}

func D2L2Regularization(w tensor.D2, lambda float64) float64 {
	sum := 0.0
	for i := range w {
		for j := range w[i] {
			ele := w[i][j]
			sum +=  ele * ele
		}
	}
	return 0.5 * lambda * sum
}

func D2L2RegularizationDerivative(w tensor.D2, lambda float64) tensor.D2 {
	grad := make(tensor.D2, len(w))
	for i := range w {
		grad[i] = D1L2RegularizationDerivative(w[i], lambda)
	}
	return grad
}