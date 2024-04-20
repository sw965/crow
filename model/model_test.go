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
	forwards := layer.D1Forwards{
		layer.NewD1AffineForward(d3Var.Param[0], d2Var.Param[0], d3Var.GetGrad()[0], d2Var.GetGrad()[0]),
		//layer.NewD1PReLUForward(&d1Var.Param[0], &d1Var.GetGrad()[0]),
		layer.NewD1PRReLUForward(&d1Var.Param[0], 0.5, 1.5, &d1Var.GetGrad()[0], &isTrain[0], r),
		//layer.NewD1LReLUForward(0.01),
		//layer.NewD1DropoutForward(0.05, &isTrain[0], r),

		layer.NewD1AffineForward(d3Var.Param[1], d2Var.Param[1], d3Var.GetGrad()[1], d2Var.GetGrad()[1]),
		//layer.NewD1PReLUForward(&d1Var.Param[1], &d1Var.GetGrad()[1]),
		layer.NewD1PRReLUForward(&d1Var.Param[1], 0.5, 1.5, &d1Var.GetGrad()[1], &isTrain[0], r),
		//layer.NewD1LReLUForward(0.01),
		//layer.NewD1DropoutForward(0.05, &isTrain[0], r),

		layer.NewD1AffineForward(d3Var.Param[2], d2Var.Param[2], d3Var.GetGrad()[2], d2Var.GetGrad()[2]),
		//layer.NewD1PReLUForward(&d1Var.Param[2], &d1Var.GetGrad()[2]),
		layer.NewD1PRReLUForward(&d1Var.Param[2], 0.5, 1.5, &d1Var.GetGrad()[2], &isTrain[0], r),
		//layer.NewD1LReLUForward(0.01),
		//layer.NewD1DropoutForward(0.05, &isTrain[0], r),
		//layer.NewD1TanhForward(),
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

	mnist, err := dataset.LoadMnist()
	if err != nil {
		panic(err)
	}

	// oldParamD1 := d1Var.Param.Clone()
	// oldParamD2 := d2Var.Param.Clone()
	// oldParamD3 := d3Var.Param.Clone()
	threshold := 8.0
	trainNum := 256000
	for i := 0; i < trainNum; i++ {
		idx := r.Intn(60000)
		model.UpdateGradAndTrain(mnist.TrainImg[idx], mnist.TrainLabel[idx], 0.01, threshold)

		// if i%16 == 0 {
		// 	model.SWA(0.9, oldParamD1, oldParamD2, oldParamD3)
		// 	oldParamD1 = d1Var.Param.Clone()
		// 	oldParamD2 = d2Var.Param.Clone()
		// 	oldParamD3 = d3Var.Param.Clone()
		// }

		// if i%1960 == 0 {
		// 	tmp := r.Intn(10000)
		// 	model.ValidateBackwardGrad(mnist.TestImg[tmp], mnist.TestLabel[tmp], threshold)
		// }

		if i%196 == 0 {
			testSize := 128
			lossSum := 0.0
			a := 0.0
			isTrain[0] = false
			for j := 0; j < testSize; j++ {
				idx := r.Intn(10000)
				_, loss, _, err := model.YAndLoss(mnist.TestImg[idx], mnist.TestLabel[idx])
				if err != nil {
					panic(err)
				}
				lossSum += loss
				acc, err := model.Accuracy(mnist.TestImg[idx], mnist.TestLabel[idx])
				if err != nil {
					panic(err)
				}
				a += acc
			}
			fmt.Println("i = ", i, "lossSum = ", lossSum)
			fmt.Println("a = ", float64(a) / float64(testSize))
			fmt.Println(d1Var.Param)
			isTrain[0] = true
		}
	}
}