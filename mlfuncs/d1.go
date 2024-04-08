package mlfuncs

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func D1MeanDerivative(x tensor.D1) tensor.D1 {
	n := len(x)
	gi := 1.0 / float64(n)
	grad := make(tensor.D1, n)
	for i := range x {
		grad[i] = gi
	}
	return grad
}

func D1VarianceDerivative(x tensor.D1) tensor.D1 {
	n := len(x)
	m := omw.Mean(x...)
	grad := make(tensor.D1, n)
	for i := range x {
		grad[i] = 2 * (x[i] - m) / float64(n)
	}
	return grad
}

func D1StandardDeviationDerivative(x tensor.D1) tensor.D1 {
	v := x.Var()
	s := math.Sqrt(v)
	vGrad := D1VarianceDerivative(x)
	n := len(x)
	grad := make(tensor.D1, n)
	for i := range x {
		grad[i] = vGrad[i] / (2 * s)
	}
	return grad
}

func D1StandardizeDerivative(x tensor.D1) tensor.D1 {
	n := len(x)
	m := omw.Mean(x...)
	std := x.Std()
	grad := make(tensor.D1, n)
	for i := range x {
		for j := range x {
			delta := 0.0
			if i == j {
				delta = 1.0
			}
			grad[j] += ((delta - 1.0/float64(n)) / std) - ((x[j] - m) * (x[i] - m) / (float64(n) * std * std * std))
		}
	}
	return grad
}

func D1Sigmoid(x tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](x, ScalarSigmoid)
}

func D1SigmoidGrad(x tensor.D1) tensor.D1 {
	return omw.MapFunc[tensor.D1](x, ScalarSigmoidGrad)
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

func D1PReLU(x tensor.D1, alpha tensor.D1) (tensor.D1, error) {
	if len(x) != len(alpha) {
		return tensor.D1{}, fmt.Errorf("xとalphaの長さが異なるため、D1PReLUを計算出来ません。")
	}

	y := make(tensor.D1, len(x))
	for i := range x {
		xi := x[i]
		if xi > 0 {
			y[i] = xi
		} else {
			y[i] = xi * alpha[i]
		}
	}
	return y, nil
}

func D1PReLUDerivative(x tensor.D1, alpha tensor.D1) (tensor.D1, tensor.D1, error) {
	if len(x) != len(alpha) {
		return tensor.D1{}, tensor.D1{}, fmt.Errorf("xとalphaの長さが異なるため、D1PReLUDerivativeを計算出来ません。")
	}

	xGrad := make(tensor.D1, len(x))
	alphaGrad := make(tensor.D1, len(alpha))

	for i := range x {
		xi := x[i]
		if xi > 0 {
			xGrad[i] = 1
			alphaGrad[i] = 0
		} else {
			xGrad[i] = alpha[i]
			alphaGrad[i] = xi
		}
	}
	return xGrad, alphaGrad, nil
}

func D1MeanSquaredError(y, t tensor.D1) (float64, error) {
	if len(y) != len(t) {
		return 0.0, fmt.Errorf("yとtの長さが異なるため、D1MeanSquaredErrorが計算出来ません。")
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
		return tensor.D1{}, fmt.Errorf("yとtの長さが異なるため、D1MeanSquaredErrorDerivativeが計算出来ません。")
	}

	n := len(y)
    grad := make(tensor.D1, n)
    for i := range y {
        grad[i] = (y[i] - t[i]) / float64(n)
    }
    return grad, nil
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

func NormalizeToProbDist(x tensor.D1) tensor.D1 {
	sum := omw.Sum(x...)
	return x.DivScalar(sum)
}

func NormalizeToProbDistDerivative(x tensor.D1) tensor.D1 {
    sum := omw.Sum(x...)
    ss := sum * sum
    grad := make(tensor.D1, len(x))
    for i := range x {
        grad[i] = (sum - x[i]) / ss
    }
    return grad
}

func D1Dropout(x tensor.D1, p float64, r *rand.Rand) (tensor.D1, []bool) {
	n := len(x)
	y := make(tensor.D1, n)
	mask := make([]bool, n)
	for i := range x {
		if r.Float64() < p {
			y[i] = 0.0
			mask[i] = true
		} else {
			y[i] = x[i] / (1 - p)
			mask[i] = false
		}
	}
	return y, mask
}

func D1DropoutDerivative(mask []bool) tensor.D1 {
	grad := make(tensor.D1, len(mask))
	for i := range mask {
		if mask[i] {
			grad[i] = 0.0
		} else {
			grad[i] = 1.0
		}
	}
	return grad
}