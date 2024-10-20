package model1d_test

import (
	"fmt"
	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model/1d"
	"github.com/sw965/crow/tensor"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
	"testing"
)

func TestModel(t *testing.T) {
	return
	r := omwrand.NewMt19937()
	xn := 784
	h1 := 128
	h2 := 32
	yn := 10

	affine, _ := model1d.NewStandardAffine(xn, h1, h2, yn, 0.0001, 64.0, r)
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

func TestLinearSum(test *testing.T) {
	r := omwrand.NewMt19937()
	xn := 10
	yn := 1
	linearSum, _ := model1d.NewStandardLinearSum(xn, 0.001, 64.0)
	x := tensor.NewD1RandUniform(xn, -5.0, 5.0, r)
	t := tensor.NewD1RandUniform(yn, 0.0, 1.0, r)
	linearSum.ValidateBackwardAndNumericalGradientDifference(x, t)
}
