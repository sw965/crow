package model_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
	"github.com/sw965/crow/tensor"
)

func TestModel(t *testing.T) {
	r := omwrand.NewMt19937()
	xn := 784
	h1 := 128
	h2 := 32
	yn := 10
	affine, _ := model.NewThreeLayerAffineParamReLUInput1DOutputSigmoid1D(xn, h1, h2, yn, 0.0001, 64.0, r)

	mnist, err := dataset.LoadFlatMnist()
	if err != nil {
		panic(err)
	}

	trainImgNum := len(mnist.TrainImg)
	testImgNum := len(mnist.TestImg)
	trainNum := 256000
	testSize := 196

	for i := 0; i < trainNum; i++ {
		idx := r.Intn(trainImgNum)
		affine.SGD(mnist.TrainImg[idx], mnist.TrainLabel[idx], 0.01)
		if i%196 == 0 {
			//affine.ValidateBackwardAndNumericalGradientDifference(mnist.TrainImg[idx], mnist.TrainLabel[idx])
			idxs := omwrand.IntsUniform(testSize, 0, testImgNum, r)
			miniBatchTestImg := omwslices.IndicesAccess(mnist.TestImg, idxs...)
			miniBatchTestLabel := omwslices.IndicesAccess(mnist.TestLabel, idxs...)
			loss, err := affine.MeanLoss(miniBatchTestImg, miniBatchTestLabel)
			if err != nil {
				panic(err)
			}
			accuracy, err := affine.Accuracy(miniBatchTestImg, miniBatchTestLabel)
			if err != nil {
				panic(err)
			}

			fmt.Println("i =", i, "loss =", loss, "accuracy =", accuracy)
		}
	}
}

func TestLinearSumGrad(test *testing.T) {
	rng := omwrand.NewMt19937()
	linear := model.NewLinearSumIdentityMSE(0.001)
	//linear := model.NewLinearSumSigmoidMSE(0.001)
	r, c := 10, 5
	min, max := -5.0, 5.0
	linear.W = tensor.NewD2RandUniform(r, c, min, max, rng)
	linear.B = tensor.NewD1RandUniform(r, min, max, rng)
	x := tensor.NewD2RandUniform(r, c, min, max, rng)
	t := tensor.NewD1RandUniform(r, min, max, rng)
	err := linear.ValidateBackwardAndNumericalGradientDifference(x, t)
	if err != nil {
		panic(err)
	}
}