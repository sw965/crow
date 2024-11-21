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
	h1 := 64
	h2 := 16
	yn := 10

	affine := model1d.NewSoftmaxAffine([]int{xn, h1, h2, yn}, 0.1, r)
	affine.GradMaxL2Norm = 0
	mnist, err := dataset.LoadFlatMnist()
	if err != nil {
		panic(err)
	}

	testImgNum := len(mnist.TestImg)
	trainNum := 256000
	testSize := 16
	trainer := model1d.NewTrainer(&affine, mnist.TrainImg, mnist.TrainLabel, 4)
	trainer.BatchSize = 512
	momentum := model1d.NewMomentum(&affine, 0.9)
	fmt.Println(momentum.Velocity1D)
	fmt.Println(momentum.Velocity2D[0][0])
	fmt.Println(momentum.Velocity3D[0][0][0])
	trainer.Optimizer = momentum.Optimizer

	for i := 0; i < trainNum; i++ {
		err := trainer.Train(&affine, 0.01, r)
		if err != nil {
			panic(err)
		}

		fmt.Println(momentum.Velocity1D)
		fmt.Println(momentum.Velocity2D[0][0])
		fmt.Println(momentum.Velocity3D[0][0][0])

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