package d1

import (
	"github.com/sw965/crow/mlfuncs/scalar"
	"github.com/sw965/omw/fn"
)

func TanhDerivative(x tensor.D1) tensor.D1 {
	return fn.Map[tensor.D1](x, scalar.TanhDerivative)
}

func TanhGrad(y tensor.D1) tensor.D1 {
	return fn.Map[tensor.D1](y, scalar.TanhGrad)
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

func SumSquaredError(y, t tensor.D1) (float64, error) {
	if len(y) != len(t) {
		return 0, fmt.Errorf("スライスの長さが一致しません")
	}
    diff := y.Sub(t)
    return 0.5 * omwmath.Sum(diff...)
}


func MeanSquaredError(y, t tensor.D1) (float64, error) {
	sse, err := y.SumSquaredError(t)
	if err != nil {
		return 0.0, err
	}
	mse := sse / float64(len(y))
	return mse, nil
}