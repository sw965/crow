package model_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model"
	"github.com/sw965/crow/layer"
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
)

func TestModel(t *testing.T) {
	r := omw.NewMt19937()
	xSize := 784
	hidden1Size := 192
	hidden2Size := 64
	ySize := 10

	isTrain := []bool{true}
	d1Var := model.D1Var{
		Param:tensor.D1{0.1, 0.1, 0.1},
	}
	d1Var.Init(0.9)

	d2Var := model.D2Var{
		Param:tensor.D2{
			make(tensor.D1, hidden1Size),
			make(tensor.D1, hidden2Size),
			make(tensor.D1, ySize),
		},
	}
	d2Var.Init(0.9)

	d3Var := model.D3Var{
		Param:tensor.D3{
			tensor.NewD2He(xSize, hidden1Size, r),
			tensor.NewD2He(hidden1Size, hidden2Size, r),
			tensor.NewD2He(hidden2Size, ySize, r),
		},
	}
	d3Var.Init(0.9)

	model := model.NewD1(&d1Var, &d2Var, &d3Var)
	model.L2NormGradClipThreshold = 8.0
	forwards := layer.D1Forwards{
		layer.NewD1AffineForward(d3Var.Param[0], d2Var.Param[0], d3Var.GetGrad()[0], d2Var.GetGrad()[0]),
		layer.NewD1PRReLUForward(&d1Var.Param[0], 0.5, 1.5, &d1Var.GetGrad()[0], &isTrain[0], r),
		layer.NewD1DropoutForward(0.05, &isTrain[0], r),

		layer.NewD1AffineForward(d3Var.Param[1], d2Var.Param[1], d3Var.GetGrad()[1], d2Var.GetGrad()[1]),
		layer.NewD1PRReLUForward(&d1Var.Param[1], 0.5, 1.5, &d1Var.GetGrad()[1], &isTrain[0], r),
		layer.NewD1DropoutForward(0.05, &isTrain[0], r),

		layer.NewD1AffineForward(d3Var.Param[2], d2Var.Param[2], d3Var.GetGrad()[2], d2Var.GetGrad()[2]),
		layer.NewD1PRReLUForward(&d1Var.Param[2], 0.5, 1.5, &d1Var.GetGrad()[2], &isTrain[0], r),
		layer.NewD1DropoutForward(0.05, &isTrain[0], r),
	}

	model.Forwards = forwards
	model.LossForward = layer.NewD1MeanSquaredErrorForward()
	lambda := 0.0001
	model.ParamLoss = func(d1 tensor.D1, _ tensor.D2, d3 tensor.D3) float64 {
		return mlfuncs.D1L2Regularization(d1, lambda) + mlfuncs.D3L2Regularization(d3, lambda)
	}
	model.ParamLossDerivative = func(d1 tensor.D1, d2 tensor.D2, d3 tensor.D3) (tensor.D1, tensor.D2, tensor.D3) {
		return mlfuncs.D1L2RegularizationDerivative(d1, lambda),
			tensor.NewD2ZerosLike(d2),
			mlfuncs.D3L2RegularizationDerivative(d3, lambda)
	}

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
		model.UpdateGradAndTrain(mnist.TrainImg[idx], mnist.TrainLabel[idx], 0.01)
		if i%196 == 0 {
			idxs := omw.RandIntns(testSize, testImgNum, r)
			miniBatchTestImg := omw.ElementsAtIndices(mnist.TestImg, idxs...)
			miniBatchTestLabel := omw.ElementsAtIndices(mnist.TestLabel, idxs...)
			isTrain[0] = false
			loss, err := model.MeanLoss(miniBatchTestImg, miniBatchTestLabel)
			if err != nil {
				panic(err)
			}
			accuracy, err := model.Accuracy(miniBatchTestImg, miniBatchTestLabel)
			if err != nil {
				panic(err)
			}

			fmt.Println("loss =", loss, "accuracy =", accuracy)
			isTrain[0] = true
		}
	}
}