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

func TestD1LinearForward(test *testing.T) {
	r := omw.NewMt19937()
	n := 10
	x := tensor.NewD1RandomUniform(n, -0.1, 0.1, r)
	w := tensor.NewD1RandomUniform(n, -0.1, 0.1, r)
	b := r.Float64()
	t := tensor.NewD1RandomUniform(1, -0.1, 0.1, r)

	loss := func(x, w tensor.D1, b float64) float64 {
		mul, err := tensor.D1Mul(x, w)
		if err != nil {
			panic(err)
		}
		y := tensor.D1{omw.Sum(mul...) + b}
		z, err := mlfuncs.D1MeanSquaredError(y, t)
		if err != nil {
			panic(err)
		}
		return z
	}

	lossX := func(x tensor.D1) float64 { return loss(x, w, b) }
	lossW := func(w tensor.D1) float64 { return loss(x, w, b) }
	lossB := func(b float64) float64 { return loss(x, w, b) }

	numGradX := mlfuncs.D1NumericalDifferentiation(x, lossX)
	numGradW := mlfuncs.D1NumericalDifferentiation(w, lossW)
	numGradB := mlfuncs.ScalarNumericalDifferentiation(b, lossB)
	gradW := tensor.NewD1ZerosLike(w)
	gradB := tensor.D1{0.0}

	forwards := layer.D1Forwards{
		layer.NewD1LinearForward(w, b, gradW, &gradB[0]),
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

	bp := layer.NewD1BackPropagator(backwards, lossBackward)
	gradX, err := bp.Run()
	if err != nil {
		panic(err)
	}

	diffNumGradX, err := tensor.D1Sub(numGradX, gradX)
	if err != nil {
		panic(err)
	}
	maxDiffNumGradX := diffNumGradX.MapFunc(math.Abs).Max()

	diffNumGradW, err := tensor.D1Sub(numGradW, gradW)
	if err != nil {
		panic(err)
	}
	maxDiffNumGradW := diffNumGradW.MapFunc(math.Abs).Max()

	diffNumGradB := math.Abs(numGradB - gradB[0])
	fmt.Println(maxDiffNumGradX, maxDiffNumGradW, diffNumGradB)
}

func TestD1AffineForward(test *testing.T) {
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
	
	bp := layer.NewD1BackPropagator(backwards, lossBackward)
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