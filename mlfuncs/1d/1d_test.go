package mlfuncs1d_test

import (
	"fmt"
	"github.com/sw965/crow/mlfuncs/1d"
	"github.com/sw965/crow/mlfuncs/scalar"
	"github.com/sw965/crow/tensor"
	omwmath "github.com/sw965/omw/math"
	omwrand "github.com/sw965/omw/math/rand"
	"math"
	"testing"
)

func TestParamReLUDerivative(test *testing.T) {
	r := omwrand.NewMt19937()
	n := 10
	min, max := -5.0, 5.0
	x := tensor.NewD1RandUniform(n, min, max, r)
	t := tensor.NewD1RandUniform(n, min, max, r)
	alpha := omwrand.Float64Uniform(min, max, r)

	loss := func(x tensor.D1, alpha float64) float64 {
		y := mlfuncs1d.LeakyReLU(x, alpha)
		ret, err := mlfuncs1d.MeanSquaredError(y, t)
		if err != nil {
			panic(err)
		}
		return ret
	}
	lossX := func(x tensor.D1) float64 { return loss(x, alpha) }
	lossAlpha := func(alpha float64) float64 { return loss(x, alpha) }

	numGradX := mlfuncs1d.NumericalDifferentiation(x, lossX)
	numGradAlpha := scalar.NumericalDifferentiation(alpha, lossAlpha)

	dLdy, err := mlfuncs1d.MeanSquaredErrorDerivative(mlfuncs1d.LeakyReLU(x, alpha), t)
	if err != nil {
		panic(err)
	}
	dydx, dydVectorizedGradAlpha := mlfuncs1d.ParamReLUDerivative(x, alpha)
	vectorizedGradAlpha, err := tensor.D1Mul(dydVectorizedGradAlpha, dLdy)
	if err != nil {
		panic(err)
	}
	gradX, err := tensor.D1Mul(dydx, dLdy)
	if err != nil {
		panic(err)
	}
	gradAlpha := omwmath.Sum(vectorizedGradAlpha...)

	diffX, err := tensor.D1Sub(numGradX, gradX)
	if err != nil {
		panic(err)
	}
	maxDiffX := omwmath.Max(diffX...)
	diffAlpha := math.Abs(numGradAlpha - gradAlpha)
	fmt.Println("maxDiffX =", maxDiffX, "diffAlpha =", diffAlpha)
}

func TestL2RegularizationDerivative(test *testing.T) {
	r := omwrand.NewMt19937()
	n := 10
	min, max := -5.0, 5.0
	w := tensor.NewD1RandUniform(n, min, max, r)
	c := 0.01

	numGrad := mlfuncs1d.NumericalDifferentiation(w, mlfuncs1d.L2Regularization(c))
	grad := mlfuncs1d.L2RegularizationDerivative(c)(w)
	diff, err := tensor.D1Sub(numGrad, grad)
	if err != nil {
		panic(err)
	}
	maxDiff := omwmath.Max(diff...)
	fmt.Println("maxDiff =", maxDiff)
}
