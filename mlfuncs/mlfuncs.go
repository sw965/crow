package mlfuncs

import (
	"fmt"
	"math"

	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func Sigmoid(x tensor.D1) tensor.D1 {
	y := make(tensor.D1, len(x))
	for i := range x {
		xi := x[i]
		y[i] = 1 / (1 + math.Exp(-xi))
	}
	return y
}

func SigmoidGrad(y tensor.D1) tensor.D1 {
	grad := make(tensor.D1, len(y))
	for i := range y {
		yi := y[i]
		grad[i] = yi * (1.0 - yi)
	}
	return grad
}

func Tanh(x tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](x, math.Tanh)
}

func TanhGrad(y tensor.D1) tensor.D1 {
	grad := make(tensor.D1, len(y))
	for i := range y {
		yi := y[i]
		grad[i] = 1.0 - (yi*yi)
	}
	return grad
}

func TanhToSigmoidScale(y tensor.D1) tensor.D1 {
	scale := make(tensor.D1, len(y))
	for i := range y {
		yi := y[i]
		scale[i] = (yi + 1.0) / 2.0
	}
	return scale
}

func ReLU(x tensor.D1) tensor.D1 {
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

func ReLUDerivative(x tensor.D1) tensor.D1 {
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

func PReLU(x tensor.D1, alpha float64) tensor.D1 {
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

func PReLUDerivative(x tensor.D1, alpha float64) (tensor.D1, float64) {
	xGrad := make(tensor.D1, len(x))
	alphaGrad := 0.0
	for i := range x {
		xi := x[i]
		if xi > 0 {
			xGrad[i] = 1
		} else {
			xGrad[i] = alpha
			alphaGrad += xi
		}
	}
	return xGrad, alphaGrad
}

func MeanSquaredError(y, t tensor.D1) (float64, error) {
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

func MeanSquaredErrorDerivative(y, t tensor.D1) (tensor.D1, error) {
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

func L2Regularization(w tensor.D2, lambda float64) float64 {
	sum := 0.0
	for i := range w {
		sum += omw.Sum(w[i]...)
	}
	return 0.5 * lambda * sum
}

func L2RegularizationDerivative(w tensor.D2, lambda float64) tensor.D2 {
	grad := make(tensor.D2, len(w))
	for i := range w {
		for j := range w[i] {
			grad[i][j] = lambda * w[i][j]
		}
	}
	return grad
}