package mlfuncs1d

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/sw965/crow/mlfuncs/scalar"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw/fn"
	omwmath "github.com/sw965/omw/math"
	omwrand "github.com/sw965/omw/math/rand"
)

func Sigmoid(x tensor.D1) tensor.D1 {
	return fn.Map[tensor.D1](x, scalar.Sigmoid)
}

func SigmoidGrad(y tensor.D1) tensor.D1 {
	return fn.Map[tensor.D1](y, scalar.SigmoidGrad)
}

func SigmoidDerivative(x tensor.D1) tensor.D1 {
	return fn.Map[tensor.D1](x, scalar.SigmoidDerivative)
}

func SigmoidToTanh(y tensor.D1) tensor.D1 {
	return fn.Map[tensor.D1](y, scalar.SigmoidToTanh)
}

func Tanh(x tensor.D1) tensor.D1 {
	return fn.Map[tensor.D1](x, math.Tanh)
}

func TanhGrad(y tensor.D1) tensor.D1 {
	return fn.Map[tensor.D1](y, scalar.TanhGrad)
}

func TanhDerivative(x tensor.D1) tensor.D1 {
	return fn.Map[tensor.D1](x, scalar.TanhDerivative)
}

func TanhToSigmoid(y tensor.D1) tensor.D1 {
	return fn.Map[tensor.D1](y, scalar.TanhToSigmoid)
}

func Softmax(x tensor.D1) tensor.D1 {
    maxX := omwmath.Max(x...) // オーバーフロー対策
    expX := make(tensor.D1, len(x))
    sumExpX := 0.0
    for i, xi := range x {
        expX[i] = math.Exp(xi - maxX)
        sumExpX += expX[i]
    }
    y := make(tensor.D1, len(x))
    for i := range expX {
        y[i] = expX[i] / sumExpX
    }
    return y
}

func SoftmaxDerivative(y, chain tensor.D1) (tensor.D1, error) {
    n := len(y)
    if n != len(chain) {
        return nil, fmt.Errorf("len(y) != len(chain)")
    }
    dx := make(tensor.D1, n)
    for i := 0; i < n; i++ {
        sum := 0.0
        for j := 0; j < n; j++ {
            if i == j {
                sum += y[i] * (1 - y[j]) * chain[j]
            } else {
                sum -= y[i] * y[j] * chain[j]
            }
        }
        dx[i] = sum
    }
    return dx, nil
}

func LinearSum(x, w tensor.D1, b float64) (float64, error) {
	hadamard, err := tensor.D1Mul(x, w)
	y := omwmath.Sum(hadamard...) + b
	return y, err
}

func LinearSumDerivative(x, w tensor.D1) (tensor.D1, tensor.D1, error) {
	n := len(x)
	gradX := make(tensor.D1, n)
	gradW := make(tensor.D1, n)
	for i := range x {
		gradX[i] = w[i]
		gradW[i] = x[i]
	}
	return gradX, gradW, nil
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

func LeakyReLU(x tensor.D1, alpha float64) tensor.D1 {
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

func LeakyReLUDerivative(x tensor.D1, alpha float64) tensor.D1 {
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

func ParamReLUDerivative(x tensor.D1, alpha float64) (tensor.D1, tensor.D1) {
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

func RandReLU(x tensor.D1, min, max float64, isTrain bool, r *rand.Rand) (tensor.D1, float64) {
	y := make(tensor.D1, len(x))
	var noise float64
	if isTrain {
		noise = omwrand.Float64Uniform(min, max, r)
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

func ParamRandReLU(x tensor.D1, alpha, min, max float64, isTrain bool, r *rand.Rand) (tensor.D1, float64) {
	y := make(tensor.D1, len(x))
	var noise float64
	if isTrain {
		noise = omwrand.Float64Uniform(min, max, r)
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

func ParamRandReLUDerivative(x tensor.D1, alpha, noise float64) (tensor.D1, tensor.D1) {
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

func Dropout(x tensor.D1, p float64, isTrain bool, r *rand.Rand) (tensor.D1, tensor.D1) {
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

func SumSquaredError(y, t tensor.D1) (float64, error) {
	if len(y) != len(t) {
		return 0.0, fmt.Errorf("len(y) != len(t) であるため、SumSquaredErrorを計算できません。")
	}
	sqSum := 0.0
	for i := range y {
		diff := y[i] - t[i]
		sqSum += (diff * diff)
	}
	return 0.5 * sqSum, nil
}

func SumSquaredErrorDerivative(y, t tensor.D1) (tensor.D1, error) {
	if len(y) != len(t) {
		return tensor.D1{}, fmt.Errorf("len(y) != len(t) であるため、SumSquaredErrorDerivativeを計算できません。")
	}
	n := len(y)
	grad := make(tensor.D1, n)
	for i := range y {
		grad[i] = y[i] - t[i]
	}
	return grad, nil
}

func MeanSquaredError(y, t tensor.D1) (float64, error) {
	sqSum, err := SumSquaredError(y, t)
	n := len(y)
	return sqSum / float64(n), err
}

func MeanSquaredErrorDerivative(y, t tensor.D1) (tensor.D1, error) {
	n := len(y)
	grad, err := SumSquaredErrorDerivative(y, t)
	grad.DivScalar(float64(n))
	return grad, err
}

func L2Regularization(c float64) func(tensor.D1) float64 {
	return func(w tensor.D1) float64 {
		sqSum := 0.0
		for _, wi := range w {
			sqSum += wi * wi
		}
		return 0.5 * c * sqSum
	}
}

func L2RegularizationDerivative(c float64) func(tensor.D1) tensor.D1 {
	return func(w tensor.D1) tensor.D1 {
		grad := make(tensor.D1, len(w))
		for i, wi := range w {
			grad[i] = c * wi
		}
		return grad
	}
}

func NumericalDifferentiation(x tensor.D1, f func(tensor.D1) float64) tensor.D1 {
	h := 0.001
	grad := make(tensor.D1, len(x))
	for i := range x {
		tmp := x[i]

		x[i] = tmp + h
		y1 := f(x)

		x[i] = tmp - h
		y2 := f(x)

		grad[i] = (y1 - y2) / (2 * h)
		x[i] = tmp
	}
	return grad
}
