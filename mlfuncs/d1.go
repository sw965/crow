package mlfuncs

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func D1Sigmoid(x tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](x, ScalarSigmoid)
}

func D1SigmoidGrad(y tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](y, ScalarSigmoidGrad)
}

func D1SigmoidToTanh(y tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](y, ScalarSigmoidToTanh)
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

func D1LinearSum(x, w tensor.D1, b float64) (float64, error) {
	hadamard, err := tensor.D1Mul(x, w)
	y := omw.Sum(hadamard...) + b
	return y, err
}

func D1LinearSumDerivative(x, w tensor.D1) (tensor.D1, tensor.D1, float64, error) {
	n := len(x)
	gradX := make(tensor.D1, n)
	gradW := make(tensor.D1, n)
	gradB := 1.0
	for i := range x {
		gradX[i] = w[i]
		gradW[i] = x[i]
	}
	return gradX, gradW, gradB, nil
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

func D1LeakyReLU(x tensor.D1, alpha float64) tensor.D1 {
	y := make(tensor.D1, len(x))
	for i := range x {
		xi := x[i]
		if xi > 0 {
			y[i] = xi
		} else {
			y[i] = alpha * xi
		}
	}
	return y
}

func D1LeakyReLUDerivative(x tensor.D1, alpha float64) tensor.D1 {
	grad := make(tensor.D1, len(x))
	for i := range x {
		xi := x[i]
		if xi > 0 {
			grad[i] = 1
		} else {
			grad[i] =  alpha
		}
	}
	return grad
}

func D1ParamReLUDerivative(x tensor.D1, alpha float64) (tensor.D1, tensor.D1) {
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

func D1RandReLU(x tensor.D1, min, max float64, isTrain bool, r *rand.Rand) (tensor.D1, float64) {
	y := make(tensor.D1, len(x))
	var noise float64
	if isTrain {
		noise = omw.RandFloat64(min, max, r)
	} else {
		noise = (min + max) / 2.0
	}
	for i := range y {
		xi := x[i]
		if xi > 0 {
			y[i] = xi
		} else {
			y[i] = noise * xi
		}
	}
	return y, noise
}

func D1ParamRandReLU(x tensor.D1, alpha, min, max float64, isTrain bool, r *rand.Rand) (tensor.D1, float64) {
	y := make(tensor.D1, len(x))
	var noise float64
	if isTrain {
		noise = omw.RandFloat64(min, max, r)
	} else {
		noise = (min + max) / 2.0
	}
	for i := range y {
		xi := x[i]
		if xi > 0 {
			y[i] = xi
		} else {
			y[i] = alpha * noise * xi
		}
	}
	return y, noise
}

func D1ParamRandReLUDerivative(x tensor.D1, alpha, noise float64) (tensor.D1, tensor.D1) {
	gradX := make(tensor.D1, len(x))
	vectorizedGradAlpha := make(tensor.D1, len(x))
	for i := range x {
		xi := x[i]
		if xi > 0 {
			gradX[i] = 1
			vectorizedGradAlpha[i] = 0
		} else {
			gradX[i] = alpha * noise
			vectorizedGradAlpha[i] = noise * xi
		}
	}
	return gradX, vectorizedGradAlpha
}

func D1Dropout(x tensor.D1, p float64, isTrain bool, r *rand.Rand) (tensor.D1, tensor.D1) {
	n := len(x)
	y := make(tensor.D1, n)
	mask := make(tensor.D1, n)
	if isTrain {
		for i := range y {
			if p > r.Float64() {
				y[i] = 0
			} else {
				y[i] = x[i]
				mask[i] = 1
			}
		}
	} else {
		q := 1.0 - p
		for i := range y {
			y[i] = q * x[i]
			mask[i] = 1
		}
	}
	return y, mask
}

func D1MeanSquaredError(y, t tensor.D1) (float64, error) {
	if len(y) != len(t) {
		return 0.0, fmt.Errorf("len(y) != len(t) であるため、MeanSquaredErrorを計算できません。")
	}

    sqSum := 0.0
    for i := range y {
        diff := y[i] - t[i]
        sqSum += (diff * diff)
    }
    n := len(y)
	mean := sqSum/float64(n)
    return 0.5*mean, nil
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

func D1L2Regularization(w tensor.D1, lambda float64) float64 {
	sqSum := 0.0
	for i := range w {
		wi := w[i]
		sqSum += wi * wi
	}
	return 0.5 * lambda * sqSum
}

func D1L2RegularizationDerivative(w tensor.D1, lambda float64) tensor.D1 {
	grad := make(tensor.D1, len(w))
	for i := range grad {
		grad[i] = lambda * w[i]
	}
	return grad
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