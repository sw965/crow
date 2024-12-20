package ml1d

import (
	"fmt"
	"math"
	"github.com/sw965/crow/ml/scalar"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw/fn"
	omwmath "github.com/sw965/omw/math"
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

func LinearSum(x, w tensor.D1, b float64) (float64, error) {
	hadamard, err := tensor.D1Mul(x, w)
	y := omwmath.Sum(hadamard...) + b
	return y, err
}

//バイアス項(b)の微分は常に1であり、連鎖律において計算する必要性がない為、計算を省く。
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
	return fn.Map[tensor.D1](x, scalar.ReLU)
}

func ReLUDerivative(x tensor.D1) tensor.D1 {
	return fn.Map[tensor.D1](x, scalar.ReLUDerivative)
}

func LeakyReLU(alpha float64) func(tensor.D1)tensor.D1 {
	return func(x tensor.D1) tensor.D1 {
		return fn.Map[tensor.D1](x, scalar.LeakyReLU(alpha))
	}
}

func LeakyReLUDerivative(alpha float64) func(tensor.D1)tensor.D1 {
	return func(x tensor.D1) tensor.D1 {
		return fn.Map[tensor.D1](x, scalar.LeakyReLUDerivative(alpha))
	}
}

func CrossEntropyError(y, t tensor.D1) (float64, error) {
    if len(y) != len(t) {
        return 0.0, fmt.Errorf("len(y) != len(t) であるため、CrossEntropyErrorを計算できません。")
    }
    loss := 0.0
	e := 0.0001
	for i := range y {
		yi := math.Max(y[i], e)
		ti := t[i]
		loss += -ti * math.Log(yi)
	}
    return loss, nil
}

//Softmaxが出力である事が前提
func CrossEntropyErrorDerivative(y, t tensor.D1) (tensor.D1, error) {
    if len(y) != len(t) {
        return nil, fmt.Errorf("len(y) != len(t) であるため、CrossEntropyErrorDerivativeを計算できません。")
    }
    grad := make(tensor.D1, len(y))
    for i := range y {
        grad[i] = y[i] - t[i]
    }
    return grad, nil
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
