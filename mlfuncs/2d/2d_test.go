package mlfuncs2d_test

import (
	"testing"
	"fmt"
	"math"
	"github.com/sw965/crow/mlfuncs/1d"
	"github.com/sw965/crow/mlfuncs/2d"
	"github.com/sw965/crow/tensor"
	orand "github.com/sw965/omw/rand"
)

func TestLinearSumDerivative(test *testing.T) {
	rng := orand.NewMt19937()
	r, c := 10, 5
	min, max := -5.0, 5.0
	x := tensor.NewD2RandUniform(r, c, min, max, rng)
	w := tensor.NewD2RandUniform(r, c, min, max, rng)
	b := tensor.NewD1RandUniform(r, min, max, rng)
	t := tensor.NewD1RandUniform(r, min, max, rng)

	lossFunc := func(x, w tensor.D2, b tensor.D1) float64 {
		y, err := mlfuncs2d.LinearSum(x, w, b)
		if err != nil {
			panic(err)
		}
		loss, err := mlfuncs1d.MeanSquaredError(y, t)
		if err != nil {
			panic(err)
		}
		return loss
	}

	xLossFunc := func(x tensor.D2) float64 { return lossFunc(x, w, b) }
	wLossFunc := func(w tensor.D2) float64 { return lossFunc(x, w, b) }
	bLossFunc := func(b tensor.D1) float64 { return lossFunc(x, w, b) }

	numGradX := mlfuncs2d.NumericalDifferentiation(x, xLossFunc)
	numGradW := mlfuncs2d.NumericalDifferentiation(w, wLossFunc)
	numGradB := mlfuncs1d.NumericalDifferentiation(b, bLossFunc)

	y, err := mlfuncs2d.LinearSum(x, w, b)
	if err != nil {
		panic(err)
	}

	lossGrad, err := mlfuncs1d.MeanSquaredErrorDerivative(y, t)
	if err != nil {
		panic(err)
	}

	gradX, gradW, gradB, err := mlfuncs2d.LinearSumDerivative(x, w)
	if err != nil {
		panic(err)
	}

	gradX.MulD1Col(lossGrad)
	gradW.MulD1Col(lossGrad)
	gradB.Mul(lossGrad)

	diffX, err := tensor.D2Sub(numGradX, gradX)
	if err != nil {
		panic(err)
	}

	diffW, err := tensor.D2Sub(numGradW, gradW)
	if err != nil {
		panic(err)
	}

	diffB, err := tensor.D1Sub(numGradB, gradB)
	if err != nil {
		panic(err)
	}

	maxDiffX := diffX.MapFunc(math.Abs).MaxRow().Max()
	maxDiffW := diffW.MapFunc(math.Abs).MaxRow().Max()
	maxDiffB := diffB.MapFunc(math.Abs).Max()
	fmt.Println(maxDiffX, maxDiffW, maxDiffB)
}