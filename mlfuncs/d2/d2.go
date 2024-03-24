package d2

import (
	"github.com/sw965/crow/mlfuncs/d1"
	"github.com/sw965/omw/fn"
)

func TanhDerivative(x tensor.D2) tensor.D2 {
	return fn.Map[tensor.D2](x, d1.TanhDerivative)
}

func TanhGrad(y tensor.D2) tensor.D2 {
	return fn.Map[tensor.D2](y, d1.TanhGrad)
}

func LeakyReLU(alpha float64) func(tensor.D2)tensor.D2 {
	return func(x tensor.D2) tensor.D2 {
		return fn.Map[tensor.D2](x, d1.LeakyReLU(alpha))
	}
}

func LeakyReLUDerivative(alpha float64) func(tensor.D2)tensor.D2 {
	return func(x tensor.D2) tensor.D2 {
		return fn.Map[tensor.D2](x, d1.LeakyReLUDerivative(alpha))
	}
}