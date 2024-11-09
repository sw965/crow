package ml2d_test

import (
	"fmt"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/ml/2d"
	"github.com/sw965/crow/tensor"
	omwrand "github.com/sw965/omw/math/rand"
	"math"
	"testing"
)

func TestLinearSumDerivative(test *testing.T) {
	rng := omwrand.NewMt19937()
	r, c := 10, 5
	min, max := -5.0, 5.0
	x := tensor.NewD2RandUniform(r, c, min, max, rng)
	w := tensor.NewD2RandUniform(r, c, min, max, rng)
	b := tensor.NewD1RandUniform(r, min, max, rng)
	t := tensor.NewD1RandUniform(r, min, max, rng)

	lossFunc := func(x, w tensor.D2, b tensor.D1) float64 {
		y, err := ml2d.LinearSum(x, w, b)
		if err != nil {
			panic(err)
		}
		loss, err := ml1d.MeanSquaredError(y, t)
		if err != nil {
			panic(err)
		}
		return loss
	}

	xLossFunc := func(x tensor.D2) float64 { return lossFunc(x, w, b) }
	wLossFunc := func(w tensor.D2) float64 { return lossFunc(x, w, b) }
	bLossFunc := func(b tensor.D1) float64 { return lossFunc(x, w, b) }

	numGradX := ml2d.NumericalDifferentiation(x, xLossFunc)
	numGradW := ml2d.NumericalDifferentiation(w, wLossFunc)
	numGradB := ml1d.NumericalDifferentiation(b, bLossFunc)

	y, err := ml2d.LinearSum(x, w, b)
	if err != nil {
		panic(err)
	}

	lossGrad, err := ml1d.MeanSquaredErrorDerivative(y, t)
	if err != nil {
		panic(err)
	}

	gradX, gradW, gradB, err := ml2d.LinearSumDerivative(x, w)
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
