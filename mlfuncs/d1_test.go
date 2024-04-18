package mlfuncs_test

import (
	"testing"
	"fmt"
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
	"math"
)

func TestD1PReLUDerivative(test *testing.T) {
	r := omw.NewMt19937()
	x := tensor.NewD1RandomUniform(10, -0.1, 0.1, r)
	t := tensor.NewD1RandomUniform(10, -0.1, 0.1, r)

	alpha := omw.RandFloat64(-0.1, 0.1, r)

	loss := func(x tensor.D1, alpha float64) float64 {
		y := mlfuncs.D1PReLU(x, alpha)
		l, err := mlfuncs.D1MeanSquaredError(y, t)
		if err != nil {
			panic(err)
		}
		return l
	}

	lossX := func(x tensor.D1) float64 { return loss(x, alpha) }
	lossAlpha := func(alpha float64) float64 { return loss(x, alpha) }
	numGradX := mlfuncs.D1NumericalDifferentiation(x, lossX)
	numGradAlpha := mlfuncs.ScalarNumericalDifferentiation(alpha, lossAlpha)

	dydx, dydVectorizedGradAlpha := mlfuncs.D1PReLUDerivative(x, alpha)
	chain, err := mlfuncs.D1MeanSquaredErrorDerivative(mlfuncs.D1LReLU(x, alpha), t)
	if err != nil {
		panic(err)
	}
 	vectorizedGradAlpha, err := tensor.D1Mul(dydVectorizedGradAlpha, chain)
	if err != nil {
		panic(err)
	}
	gradX, err := tensor.D1Mul(dydx, chain)
	if err != nil {
		panic(err)
	}
	gradAlpha := omw.Sum(vectorizedGradAlpha...)

	diffErrX, err := tensor.D1Sub(numGradX, gradX)
	if err != nil {
		panic(err)
	}
	maxDiffErrX := omw.Max(diffErrX...)
	diffErrAlpha := math.Abs(numGradAlpha-gradAlpha)
	fmt.Println("maxDiffErrX =", maxDiffErrX, "diffErrAlpha =", diffErrAlpha)
}

func TestD1Dropout(test *testing.T) {

}

func TestD1L2RegularizationDerivative(test *testing.T) {
	r := omw.NewMt19937()
	n := 10
	w := tensor.NewD1RandomUniform(n, -1.0, 1.0, r)
	lambda := 0.01
	loss := func(w tensor.D1) float64 {
		y := mlfuncs.D1L2Regularization(w, lambda)
		return y
	}
	numGrad := mlfuncs.D1NumericalDifferentiation(w, loss)
	grad := mlfuncs.D1L2RegularizationDerivative(w, lambda)

	diffErr, err := tensor.D1Sub(numGrad, grad)
	if err != nil {
		panic(err)
	}
	maxDiffErr := omw.Max(diffErr...)
	fmt.Println("maxDiffErr =", maxDiffErr)
}