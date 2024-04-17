package model

import (
	"fmt"
	"math"
	"github.com/sw965/crow/layer"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
)

type D1 struct {
	PerLayerD1Var PerLayerD1Var
	PerLayerD2Var PerLayerD2Var
	PerLayerD3Var PerLayerD3Var

	Forwards layer.D1Forwards
	LossForward layer.D1LossForward
}

func (model *D1) Init(lossForward layer.D1LossForward) {
	model.PerLayerD1Var.Init()
	model.PerLayerD2Var.Init()
	model.PerLayerD3Var.Init()
	model.LossForward = lossForward
}

func (model *D1) Predict(x tensor.D1) (tensor.D1, layer.D1Backwards, error) {
	y, backwards, err := model.Forwards.Run(x)
	return y, backwards, err
}

func (model *D1) YAndLoss(x, t tensor.D1) (tensor.D1, float64, *layer.D1BackPropagator, error) {
	y, backwards, err := model.Predict(x)
	loss, lossBackward, err := model.LossForward(y, t)
	// loss += model.PerLayerD1Var.L2Regularization()
	// loss += model.PerLayerD2Var.L2Regularization()
	// loss += model.PerLayerD3Var.L2Regularization()
	bp := layer.D1BackPropagator{Backwards:backwards, LossBackward:lossBackward}
	return y, loss, &bp, err
}

func (model *D1) Accuracy(x, t tensor.D1) (float64, error) {
	y, _, err := model.Predict(x)
	if err != nil {
		return 0.0, err
	}

	maxY := y[0]
	maxYIdx := 0
	for i, yi := range y[1:] {
		if yi > maxY {
			maxYIdx = (i+1)
		}
	}

	maxT := t[0]
	maxTIdx := 0

	for i, ti := range t[1:] {
		if ti > maxT {
			maxTIdx = (i+1)
		}
	}

	if maxYIdx == maxTIdx {
		return 1.0, nil
	} else {
		return 0.0, nil
	}
}

func (model *D1) UpdateGrad(x, t tensor.D1) error {
	_, _, bp, err := model.YAndLoss(x, t)
	if err != nil {
		return err
	}
	_, err = bp.Run()
	// model.PerLayerD1Var.AddL2RegularizationGrad()
	// model.PerLayerD2Var.AddL2RegularizationGrad()
	model.PerLayerD3Var.AddL2RegularizationGrad()
	return err
}

func (model *D1) Train(lr float64) {
	model.PerLayerD1Var.Train(lr)
	model.PerLayerD2Var.Train(lr)
	model.PerLayerD3Var.Train(lr)
}

func (model *D1) UpdateGradAndTrain(x, t tensor.D1, lr float64) {
	model.UpdateGrad(x, t)
	model.Train(lr)
}

func (model *D1) ValidateBackwardGrad(x, t tensor.D1) {
	lossD1 := func(_ tensor.D1) float64 {
		_, loss, _, err := model.YAndLoss(x, t)
		if err != nil {
			panic(err)
		}
		return loss
	}

	lossD2 := func(_ tensor.D2) float64 {
		_, loss, _, err := model.YAndLoss(x, t)
		if err != nil {
			panic(err)
		}
		return loss
	}

	lossD3 := func(d2Params tensor.D3) float64 {
		_, loss, _, err := model.YAndLoss(x, t)
		if err != nil {
			panic(err)
		}
		return loss
	} 

	numGradD1 := mlfuncs.D1NumericalDifferentiation(model.PerLayerD1Var.Param, lossD1)
	numGradD2 := mlfuncs.D2NumericalDifferentiation(model.PerLayerD2Var.Param, lossD2)
	numGradD3 := mlfuncs.D3NumericalDifferentiation(model.PerLayerD3Var.Param, lossD3)
	model.UpdateGrad(x, t)

	maxDiffD1 := 0.0
	for i := range numGradD1 {
		diff := math.Abs(model.PerLayerD1Var.Grad[i] - numGradD1[i])
		if diff > maxDiffD1 {
			maxDiffD1 = diff
		}
	}

	maxDiffD2 := 0.0
	for i := range numGradD2 {
		for j := range numGradD2[i] {
			diff := math.Abs(model.PerLayerD2Var.Grad[i][j] - numGradD2[i][j])
			if diff > maxDiffD1 {
				maxDiffD2 = diff
			}
		}
	}

	maxDiffD3 := 0.0
	for i := range numGradD3 {
		for j := range numGradD3[i] {
			for k := range numGradD3[i][j] {
				diff := math.Abs(model.PerLayerD3Var.Grad[i][j][k] - numGradD3[i][j][k])
				if diff > maxDiffD3 {
					maxDiffD3 = diff
				}
			}
		}
	}
	fmt.Println(maxDiffD1, maxDiffD2, maxDiffD3)
}