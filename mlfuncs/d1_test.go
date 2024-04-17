package mlfuncs_test

import (
	"testing"
	"fmt"
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
)

func TestD1PReLUDerivative(test *testing.T) {
	r := omw.NewMt19937()
	x := tensor.NewD1RandomUniform(10, -0.1, 0.1, r)
	t := tensor.NewD1RandomUniform(10, -0.1, 0.1, r)

	alpha := omw.RandFloat64(-0.1, 0.1, r)

	loss := func(alpha float64) float64 {
		y := mlfuncs.D1LReLU(x, alpha)
		l, err := mlfuncs.D1MeanSquaredError(y, t)
		if err != nil {
			panic(err)
		}
		return l
	}

	numGradAlpha := mlfuncs.ScalarNumericalDifferentiation(alpha, loss)
	_, dydVectorizedGradAlpha := mlfuncs.D1PReLUDerivative(x, alpha)
	chain, err := mlfuncs.D1MeanSquaredErrorDerivative(mlfuncs.D1LReLU(x, alpha), t)
	if err != nil {
		panic(err)
	}

 	vectorizedGradAlpha, err := tensor.D1Mul(dydVectorizedGradAlpha, chain)
	if err != nil {
		panic(err)
	}
	fmt.Println(numGradAlpha, omw.Sum(vectorizedGradAlpha...))
}