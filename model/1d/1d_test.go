package model1d_test

import (
	"fmt"
	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model/1d"
	//"github.com/sw965/crow/tensor"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
	"testing"
)

func TestModel(t *testing.T) {
	//return
	r := omwrand.NewMt19937()
	xn := 784
	h1 := 128
	h2 := 32
	yn := 10

	affine := model1d.NewSoftmaxAffine([]int{xn, h1, h2, yn}, 0.1, r)
	affine.GradMaxL2Norm = 50
	mnist, err := dataset.LoadFlatMnist()
	if err != nil {
		panic(err)
	}

	trainImgNum := len(mnist.TrainImg)
	testImgNum := len(mnist.TestImg)
	trainNum := 256000
	testSize := 196

	for i := 0; i < trainNum; i++ {
		idxs := omwrand.Ints(128, 0, trainImgNum, r)
		gradCacher, err := affine.ComputeGrad(omwslices.ElementsByIndices(mnist.TrainImg, idxs...), omwslices.ElementsByIndices(mnist.TrainLabel, idxs...), 4)
		if err != nil {
			panic(err)
		}
		err = affine.SGD(&gradCacher, 0.01)
		if err != nil {
			panic(err)
		}
		if i%196 == 0 {
			//affine.ValidateBackwardAndNumericalGradientDifference(mnist.TrainImg[idx], mnist.TrainLabel[idx])
			idxs := omwrand.Ints(testSize, 0, testImgNum, r)
			miniBatchTestImg := omwslices.ElementsByIndices(mnist.TestImg, idxs...)
			miniBatchTestLabel := omwslices.ElementsByIndices(mnist.TestLabel, idxs...)
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