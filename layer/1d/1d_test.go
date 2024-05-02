package layer1d_test

import (
	"testing"
	"fmt"
	"math"
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/layer/1d"
	"github.com/sw965/crow/mlfuncs/1d"
	"github.com/sw965/crow/mlfuncs/2d"
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
		dot := tensor.D2{x}.DotProduct(w)
		y, err := tensor.D1Add(dot[0], b)
		if err != nil {
			panic(err)
		}
		l, err := mlfuncs1d.MeanSquaredError(y, t)
		if err != nil {
			panic(err)
		}
		return l
	}

	lossX := func(x tensor.D1) float64 { return loss(x, w, b) }
	lossW := func(w tensor.D2) float64 { return loss(x, w, b) }
	lossB := func(b tensor.D1) float64 { return loss(x, w, b) }
	numGradX := mlfuncs1d.NumericalDifferentiation(x, lossX)
	numGradW := mlfuncs2d.NumericalDifferentiation(w, lossW)
	numGradB := mlfuncs1d.NumericalDifferentiation(b, lossB)

	forwards := layer1d.Forwards{
		layer1d.NewAffineForward(w, b, gradW, gradB),
	}
	y, backwards, err := forwards.Run(x)
	if err != nil {
		panic(err)
	}
	lossForward := layer1d.NewMeanSquaredErrorForward()
	_, lossBackward, err := lossForward(y, t)
	if err != nil {
		panic(err)
	}
	
	bp := layer1d.NewBackPropagator(backwards, lossBackward)
	gradX, err := bp.Run()
	if err != nil {
		panic(err)
	}

	diffNumGradX, err := tensor.D1Sub(numGradX, gradX)
	if err != nil {
		panic(err)
	}
	maxDiffNumGradX := diffNumGradX.MapFunc(math.Abs).Max()

	diffNumGradW, err := tensor.D2Sub(numGradW, gradW)
	if err != nil {
		panic(err)
	}
	maxDiffNumGradW := diffNumGradW.MapFunc(math.Abs).MaxAxisRow().Max()

	diffNumGradB, err := tensor.D1Sub(numGradB, gradB)
	if err != nil {
		panic(err)
	}
	maxDiffNumGradB := diffNumGradB.MapFunc(math.Abs).Max()

	fmt.Println(maxDiffNumGradX, maxDiffNumGradW, maxDiffNumGradB)
}