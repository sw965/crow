package layer_test

import (
	"testing"
	"fmt"
	"math"
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/layer"
	"github.com/sw965/crow/mlfuncs"
)

func TestAffineForward(test *testing.T) {
	random := omw.NewMt19937()
	r := 10
	c := 5

	x := tensor.NewD1RandomUniform(r, -1.0, 1.0, random)
	w := tensor.NewD2He(r, c, random)
	b := tensor.NewD1RandomUniform(c, -1.0, 1.0, random)
	gradW := tensor.NewD2ZerosLike(w)
	gradB := make(tensor.D1, len(b))
	t := tensor.NewD1RandomUniform(c, -1.0, 1.0, random)

	loss := func(x tensor.D1, w tensor.D2, b tensor.D1) float64 {
		dot, err := tensor.D2{x}.DotProduct(w)
		if err != nil {
			panic(err)
		}
		y, err := tensor.D1Add(dot[0], b)
		if err != nil {
			panic(err)
		}
		l, err := mlfuncs.D1MeanSquaredError(y, t)
		if err != nil {
			panic(err)
		}
		return l
	}

	lossX := func(x tensor.D1) float64 { return loss(x, w, b) }
	lossW := func(w tensor.D2) float64 { return loss(x, w, b) }
	lossB := func(b tensor.D1) float64 { return loss(x, w, b) }
	numGradX := mlfuncs.D1NumericalDifferentiation(x, lossX)
	numGradW := mlfuncs.D2NumericalDifferentiation(w, lossW)
	numGradB := mlfuncs.D1NumericalDifferentiation(b, lossB)

	forwards := layer.D1Forwards{
		layer.NewD1AffineForward(w, b, gradW, gradB),
	}
	y, backwards, err := forwards.Run(x)
	if err != nil {
		panic(err)
	}
	lossForward := layer.NewD1MeanSquaredErrorForward()
	_, lossBackward, err := lossForward(y, t)
	if err != nil {
		panic(err)
	}
	
	bp := layer.D1BackPropagator{Backwards:backwards, LossBackward:lossBackward}
	gradX, err := bp.Run()
	if err != nil {
		panic(err)
	}

	maxDiffNumGradX := 0.0
	for i := range numGradX {
		diff := math.Abs(numGradX[i] - gradX[i])
		if diff > maxDiffNumGradX {
			maxDiffNumGradX = diff
		}
	}

	maxDiffNumGradW := 0.0
	for i := range numGradW {
		for j := range numGradW[i] {
			diff := math.Abs(numGradW[i][j] - gradW[i][j])
			if diff > maxDiffNumGradW {
				maxDiffNumGradW = diff
			}
		}
	}

	maxDiffNumGradB := 0.0
	for i := range numGradB {
		diff := math.Abs(numGradB[i] - gradB[i])
		if diff > maxDiffNumGradB {
			maxDiffNumGradB = diff
		}
	}

	fmt.Println(maxDiffNumGradX, maxDiffNumGradW, maxDiffNumGradB)
}