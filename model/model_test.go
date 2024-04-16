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
	hidden1Size := 64
	hidden2Size := 16
	outputSize := 10

	perLayerScalarVar := model.PerLayerScalarVar{
		Param:tensor.D1{0.01, 0.01, 0.01},
	}
	perLayerD1Var := model.PerLayerD1Var{
		Param:tensor.D2{
			make(tensor.D1, hidden1Size),
			make(tensor.D1, hidden2Size),
			make(tensor.D1, outputSize),
		},
	}
	perLayerD2Var:= model.PerLayerD2Var{
		Param:tensor.D3{
			tensor.NewD2He(inputSize, hidden1Size, r),
			tensor.NewD2He(hidden1Size, hidden2Size, r),
			tensor.NewD2He(hidden2Size, outputSize, r),
		},
	}

	model := model.D1{
		PerLayerScalarVar:perLayerScalarVar,
		PerLayerD1Var:perLayerD1Var,
		PerLayerD2Var:perLayerD2Var,
	}

	model.Init(layer.NewD1MeanSquaredErrorForward())

	forwards := layer.D1Forwards{
		layer.NewD1AffineForward(perLayerD2Var.Param[0], perLayerD1Var.Param[0], model.PerLayerD2Var.Grad[0], model.PerLayerD1Var.Grad[0]),
		//layer.NewD1PReLUForward(&perLayerScalarVar.Param[0], &model.PerLayerScalarVar.Grad[0]),
		layer.NewD1ReLUForward(),
		//layer.NewD1SigmoidForward(),

		layer.NewD1AffineForward(perLayerD2Var.Param[1], perLayerD1Var.Param[1], model.PerLayerD2Var.Grad[1], model.PerLayerD1Var.Grad[1]),
		//layer.NewD1PReLUForward(&perLayerScalarVar.Param[1], &model.PerLayerScalarVar.Grad[1]),
		layer.NewD1ReLUForward(),
		//layer.NewD1SigmoidForward(),

		layer.NewD1AffineForward(perLayerD2Var.Param[2], perLayerD1Var.Param[2], model.PerLayerD2Var.Grad[2], model.PerLayerD1Var.Grad[2]),
		//layer.NewD1PReLUForward(&perLayerScalarVar.Param[2], &model.PerLayerScalarVar.Grad[2]),
		layer.NewD1ReLUForward(),
		//layer.NewD1SigmoidForward(),
	}

	model.Forwards = forwards

	mnist, err := dataset.LoadMnist()
	if err != nil {
		panic(err)
	}

	trainNum := 128000
	for i := 0; i < trainNum; i++ {
		idx := r.Intn(60000)
		model.UpdateGradAndTrain(mnist.TrainImg[idx], mnist.TrainLabel[idx], 0.01)

		if i%196 == 0 {
			tmp := r.Intn(10000)
			err := model.ValidateBackwardGrad(mnist.TestImg[tmp], mnist.TestLabel[tmp], 0.001)
			if err != nil {
				panic(err)
			}

			testSize := 128
			lossSum := 0.0
			a := 0.0
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
			fmt.Println("i = ", i, "lossSum = ", lossSum, perLayerScalarVar.Param, model.PerLayerScalarVar.Grad)
			fmt.Println("a = ", float64(a) / float64(testSize))
		}
	}
}