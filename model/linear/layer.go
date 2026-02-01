package linear

import (
	"math"
	"slices"
)

type OutputLayer struct {
	Func       func([]float32) []float32
	Derivative func([]float32) []float32
}

func NewIdentityLayer() OutputLayer {
	f := func(u []float32) []float32 {
		return slices.Clone(u)
	}

	d := func(y []float32) []float32 {
		grad := make([]float32, len(y))
		for i := range grad {
			grad[i] = 1.0
		}
		return grad
	}

	return OutputLayer{
		Func:       f,
		Derivative: d,
	}
}

func NewSigmoidLayer() OutputLayer {
	f := func(u []float32) []float32 {
		y := make([]float32, len(u))
		for i := range y {
			y[i] = 1.0 / (1.0 + float32(math.Exp(float64(-u[i]))))
		}
		return y
	}

	d := func(y []float32) []float32 {
		grad := make([]float32, len(y))
		for i := range grad {
			yi := y[i]
			grad[i] = (1.0 - yi) * yi
		}
		return grad
	}

	return OutputLayer{
		Func:       f,
		Derivative: d,
	}
}

func NewSoftmaxLayerForCrossEntropy() OutputLayer {
	f := func(u []float32) []float32 {
		n := len(u)
		maxU := slices.Max(u)
		expU := make([]float32, n)
		sumExpU := float32(0.0)
		for i := range u {
			expU[i] = float32(math.Exp(float64(u[i] - maxU)))
			sumExpU += expU[i]
		}
		y := make([]float32, n)
		for i := range expU {
			y[i] = expU[i] / sumExpU
		}
		return y
	}

	return OutputLayer{
		Func:       f,
		Derivative: nil, // CrossEntropyが前提であれば、連鎖律をそのまま通せばいい
	}
}

type PredictLossLayer struct {
	Func       func([]float32, []float32) float32
	Derivative func([]float32, []float32) []float32
}

func NewMSELossLayer() PredictLossLayer {
	f := func(y, t []float32) float32 {
		sqSum := float32(0.0)
		for i := range y {
			diff := y[i] - t[i]
			sqSum += (diff * diff)
		}
		return 0.5 * sqSum
	}

	d := func(y, t []float32) []float32 {
		grad := make([]float32, len(y))
		for i := range grad {
			grad[i] = y[i] - t[i]
		}
		return grad
	}
	return PredictLossLayer{
		Func:       f,
		Derivative: d,
	}
}

// 100万分の1
const minProb float32 = 1.0 / 1000000.0

func NewCrossEntropyLossLayerForSoftmax() PredictLossLayer {
	f := func(y, t []float32) float32 {
		loss := float32(0.0)
		for i := range y {
			yi := y[i]
			if yi < minProb {
				yi = minProb
			}
			loss += -t[i] * float32(math.Log(float64(yi)))
		}
		return loss
	}

	d := func(y, t []float32) []float32 {
		grad := make([]float32, len(y))
		for i := range grad {
			grad[i] = y[i] - t[i]
		}
		return grad
	}

	return PredictLossLayer{Func: f, Derivative: d}
}
