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

	testImgNum := len(mnist.TestImg)
	trainNum := 256000
	testSize := 196
	momentum := model1d.NewMomentum(&affine, 0.9)
	trainer := model1d.Trainer{
		TeacherXs:mnist.TrainImg,
		TeacherYs:mnist.TrainLabel,
		Optimizer:momentum.Optimizer,
		BatchSize:512,
		Epoch:1,
	}

	for i := 0; i < trainNum; i++ {
		err := trainer.Train(&affine, 0.01, 4, r)
		if err != nil {
			panic(err)
		}

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