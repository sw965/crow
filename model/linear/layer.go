package linear

import (
	"github.com/chewxy/math32"
	omath "github.com/sw965/omw/math"
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
			y[i] = 1.0 / (1.0 + math32.Exp(-u[i]))
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

func NewSoftmaxLayer() OutputLayer {
	f := func(u []float32) []float32 {
		un := len(u)
		maxU := omath.Max(u...) // オーバーフロー対策
		expU := make([]float32, un)
		sumExpU := float32(0.0)
		for i := range u {
			expU[i] = math32.Exp(u[i] - maxU)
			sumExpU += expU[i]
		}

		y := make([]float32, un)
		for i := range expU {
			y[i] = expU[i] / sumExpU
		}
		return y
	}

	return OutputLayer{
		Func: f,
		/*
			CrossEntropyLossを使うことを前提としている為、導関数は不要
			連鎖律で伝ってくる勾配をそのまま通せばいい
		*/
		Derivative: nil,
	}
}

type PredictLossLayer struct {
	Func       func([]float32, []float32) float32
	Derivative func([]float32, []float32) []float32
}

func NewMSELoss() PredictLossLayer {
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

func NewCrossEntropyLossLayer() PredictLossLayer {
	const eps float32 = 0.0001
	f := func(y, t []float32) float32 {
		loss := float32(0.0)
		for i := range y {
			// ye := omath.Max(y[i], 0.0001)
			// ye = omath.Min(ye, 0.9999)
			loss += -t[i] * math32.Log(y[i]+eps)
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

	return PredictLossLayer{
		Func:       f,
		Derivative: d,
	}
}
