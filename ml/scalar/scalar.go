package scalar

import (
	"math"
)

func Sigmoid(x float32) float32 {
	y := 1.0 / (1 + math.Exp(float64(-x)))
	return float32(y)
}

func SigmoidGrad(y float32) float32 {
	return y * (1.0 - y)
}

func SigmoidDerivative(x float32) float32 {
	y := Sigmoid(x)
	return SigmoidGrad(y)
}

func ReLU(x float32) float32 {
	if x > 0 {
		return x
	} else {
		return 0
	}
}

func ReLUDerivative(x float32) float32 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}

func LeakyReLU(alpha float32) func(float32) float32 {
	return func(x float32) float32 {
		if x > 0 {
			return x
		} else {
			return x * alpha
		}
	}
}

func LeakyReLUDerivative(alpha float32) func(float32) float32 {
	return func(x float32) float32 {
		if x > 0 {
			return 1
		} else {
			return alpha
		}
	}
}