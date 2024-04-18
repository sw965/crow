package model_test

import (
	"testing"
	"fmt"

	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model"
	"github.com/sw965/crow/layer"
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

func TestModel(t *testing.T) {
	r := omw.NewMt19937()
	inputSize := 784
	hidden1Size := 196
	hidden2Size := 64
	outputSize := 10

	isTrain := true
	perLayerD1Var := model.PerLayerD1Var{
		Param:tensor.D1{0.01, 0.01, 0.01},
	}
	perLayerD1Var.Init()

	perLayerD2Var := model.PerLayerD2Var{
		Param:tensor.D2{
			make(tensor.D1, hidden1Size),
			make(tensor.D1, hidden2Size),
			make(tensor.D1, outputSize),
		},
	}
	perLayerD2Var.Init()

	perLayerD3Var:= model.PerLayerD3Var{
		Param:tensor.D3{
			tensor.NewD2He(inputSize, hidden1Size, r),
			tensor.NewD2He(hidden1Size, hidden2Size, r),
			tensor.NewD2He(hidden2Size, outputSize, r),
		},
	}
	perLayerD3Var.Init()

	model := model.NewD1(&perLayerD1Var, &perLayerD2Var, &perLayerD3Var)
	forwards := layer.D1Forwards{
		layer.NewD1AffineForward(perLayerD3Var.Param[0], perLayerD2Var.Param[0], perLayerD3Var.GetGrad()[0], perLayerD2Var.GetGrad()[0]),
		layer.NewD1PReLUForward(&perLayerD1Var.Param[0], &perLayerD1Var.GetGrad()[0]),
		//layer.NewD1DropoutForward(0.1, &isTrain, r),

		layer.NewD1AffineForward(perLayerD3Var.Param[1], perLayerD2Var.Param[1], perLayerD3Var.GetGrad()[1], perLayerD2Var.GetGrad()[1]),
		layer.NewD1PReLUForward(&perLayerD1Var.Param[1], &perLayerD1Var.GetGrad()[1]),
		//layer.NewD1DropoutForward(0.1, &isTrain, r),

		layer.NewD1AffineForward(perLayerD3Var.Param[2], perLayerD2Var.Param[2], perLayerD3Var.GetGrad()[2], perLayerD2Var.GetGrad()[2]),
		layer.NewD1PReLUForward(&perLayerD1Var.Param[2], &perLayerD1Var.GetGrad()[2]),
		//layer.NewD1DropoutForward(0.1, &isTrain, r),
	}
	model.Forwards = forwards
	model.LossForward = layer.NewD1MeanSquaredErrorForward()

	mnist, err := dataset.LoadMnist()
	if err != nil {
		panic(err)
	}

	oldParamD1 := perLayerD1Var.Param.Clone()
	oldParamD2 := perLayerD2Var.Param.Clone()
	oldParamD3 := perLayerD3Var.Param.Clone()

	trainNum := 256000
	for i := 0; i < trainNum; i++ {
		idx := r.Intn(60000)
		model.UpdateGradAndTrain(mnist.TrainImg[idx], mnist.TrainLabel[idx], 0.01)

		if i%8 == 0 {
			model.SWA(0.5, oldParamD1, oldParamD2, oldParamD3)
			oldParamD1 = perLayerD1Var.Param.Clone()
			oldParamD2 = perLayerD2Var.Param.Clone()
			oldParamD3 = perLayerD3Var.Param.Clone()
		}

		// if i%1960 == 0 {
		// 	tmp := r.Intn(10000)
		// 	model.ValidateBackwardGrad(mnist.TestImg[tmp], mnist.TestLabel[tmp])
		// }

		if i%196 == 0 {
			testSize := 128
			lossSum := 0.0
			a := 0.0
			isTrain = false
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
			isTrain = true
		}
	}
}