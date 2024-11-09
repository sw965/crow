package layer1d_test

import (
	"fmt"
	"github.com/sw965/crow/layer/1d"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/ml/2d"
	"github.com/sw965/crow/tensor"
	omwrand "github.com/sw965/omw/math/rand"
	"math"
	"testing"
)

func TestAffineForward(test *testing.T) {
	rg := omwrand.NewMt19937()
	r := 10
	c := 5

	x := tensor.NewD1RandUniform(r, -1.0, 1.0, rg)
	w := tensor.NewD2He(r, c, rg)
	b := tensor.NewD1RandUniform(c, -1.0, 1.0, rg)
	gradW := tensor.NewD2ZerosLike(w)
	gradB := make(tensor.D1, len(b))
	t := tensor.NewD1RandUniform(c, -1.0, 1.0, rg)

	loss := func(x tensor.D1, w tensor.D2, b tensor.D1) float64 {
		dot := tensor.D2{x}.DotProduct(w)
		y, err := tensor.D1Add(dot[0], b)
		if err != nil {
			panic(err)
		}
		l, err := ml1d.MeanSquaredError(y, t)
		if err != nil {
			panic(err)
		}
		return l
	}

	lossX := func(x tensor.D1) float64 { return loss(x, w, b) }
	lossW := func(w tensor.D2) float64 { return loss(x, w, b) }
	lossB := func(b tensor.D1) float64 { return loss(x, w, b) }

	numGradX := ml1d.NumericalDifferentiation(x, lossX)
	numGradW := ml2d.NumericalDifferentiation(w, lossW)
	numGradB := ml1d.NumericalDifferentiation(b, lossB)

	forwards := layer1d.Forwards{
		layer1d.NewAffineForward(w, b, gradW, gradB),
	}
	y, backwards, err := forwards.Propagate(x)
	if err != nil {
		panic(err)
	}

	chain, err := ml1d.MeanSquaredErrorDerivative(y, t)
	if err != nil {
		panic(err)
	}

	gradX, err := backwards.Propagate(chain)
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
	maxDiffNumGradW := diffNumGradW.MapFunc(math.Abs).MaxRow().Max()

	diffNumGradB, err := tensor.D1Sub(numGradB, gradB)
	if err != nil {
		panic(err)
	}
	maxDiffNumGradB := diffNumGradB.MapFunc(math.Abs).Max()

	fmt.Println(maxDiffNumGradX, maxDiffNumGradW, maxDiffNumGradB)
}
