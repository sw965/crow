package mlfuncs

import (
	"fmt"
	"math"
	"github.com/sw965/crow/tensor"
)

func ScalarTanhGrad(y float64) float64 {
	return 1.0 - (y*y)
}

func ScalarTanhToSigmoidScale(y float64) float64 {
	return (y + 1.0) / 2.0
}

func D1Tanh(x tensor.D1) tensor.D1 {
	y := make(tensor.D1, len(x))
	for i := range y {
		y[i] = math.Tanh(x[i])
	}
	return y
}

func D1TanhGrad(y tensor.D1) tensor.D1 {
	grad := make(tensor.D1, len(y))
	for i := range y {
		yi := y[i]
		grad[i] = 1.0 - (yi*yi) 
	}
	return grad
}

func D1TanhToSigmoidScale(y tensor.D1) tensor.D1 {
	scale := make(tensor.D1, len(y))
	for i := range y {
		scale[i] = (y[i]+ 1.0) / 2.0
	}
	return scale
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

func D1L2Loss(w tensor.D1, lambda float64) float64 {
	sum := 0.0
	for _, ele := range w {
		sum += ele * ele
	}
	return 0.5 * lambda * sum
}

func D1L2LossDerivative(w tensor.D1, lambda float64) tensor.D1 {
	grad := make(tensor.D1, len(w))
	for i := range w {
		grad[i] = lambda * w[i]
	}
	return grad
}

func D2L2Loss(w tensor.D2, lambda float64) float64 {
	sum := 0.0
	for i := range w {
		for j := range w[i] {
			ele := w[i][j]
			sum +=  ele * ele
		}
	}
	return 0.5 * lambda * sum
}

func D2L2LossDerivative(w tensor.D2, lambda float64) tensor.D2 {
	grad := make(tensor.D2, len(w))
	for i := range w {
		grad[i] = D1L2LossDerivative(w[i], lambda)
	}
	return grad
}