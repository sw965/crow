package model_test

import (
	"testing"
	"github.com/sw965/crow/model"
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func TestL(test *testing.T) {
	rng := omw.NewMt19937()
	linear := model.NewD2LinearSumTanhMSE(0.001)
	r, c := 10, 5
	w := tensor.NewD2Zeros(r, c)
	b := tensor.NewD1Zeros(r)
	linear.SetParam(w, b)
	x := tensor.NewD2RandomUniform(r, c, -0.1, 0.1, rng)
	t := tensor.NewD1RandomUniform(r, -0.1, 0.1, rng)
	err := linear.ValidateBackwardAndNumericalGradientDifference(x, t)
	if err != nil {
		panic(err)
	}
}