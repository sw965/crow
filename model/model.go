package model

import (
	"fmt"
	"math"
	"github.com/sw965/crow/layer"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
)

type D1 struct {
	perLayerD1Var PerLayerD1Var
	perLayerD2Var PerLayerD2Var
	perLayerD3Var PerLayerD3Var

	Forwards layer.D1Forwards
	LossForward layer.D1LossForward

	L2RegularizationD1Param float64
	L2RegularizationD2Param float64
	L2RegularizationD3Param float64
}

func NewD1(perLayerD1Var *PerLayerD1Var, perLayerD2Var *PerLayerD2Var, perLayerD3Var *PerLayerD3Var) D1 {
	return D1{perLayerD1Var:*perLayerD1Var, perLayerD2Var:*perLayerD2Var, perLayerD3Var:*perLayerD3Var}
}

func (model *D1) Predict(x tensor.D1) (tensor.D1, layer.D1Backwards, error) {
	y, backwards, err := model.Forwards.Run(x)
	return y, backwards, err
}

func (model *D1) YAndLoss(x, t tensor.D1) (tensor.D1, float64, *layer.D1BackPropagator, error) {
	y, backwards, err := model.Predict(x)
	loss, lossBackward, err := model.LossForward(y, t)
	loss += model.perLayerD1Var.L2Regularization(model.L2RegularizationD1Param)
	loss += model.perLayerD2Var.L2Regularization(model.L2RegularizationD2Param)
	loss += model.perLayerD3Var.L2Regularization(model.L2RegularizationD3Param)
	bp := layer.NewD1BackPropagator(backwards, lossBackward)
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
			maxY = yi
			maxYIdx = (i+1)
		}
	}

	maxT := t[0]
	maxTIdx := 0

	for i, ti := range t[1:] {
		if ti > maxT {
			maxT = ti
			maxTIdx = (i+1)
		}
	}

	if maxYIdx == maxTIdx {
		return 1.0, nil
	} else {
		return 0.0, nil
	}
}

func (model *D1) SWA(ratio float64, oldParamD1 tensor.D1, oldParamD2 tensor.D2, oldParamD3 tensor.D3) error {
	model.perLayerD1Var.Param.MulScalar(ratio)
	model.perLayerD2Var.Param.MulScalar(ratio)
	model.perLayerD3Var.Param.MulScalar(ratio)

	oldRatio := 1.0 - ratio
	scaledOldParamD1 := tensor.D1MulScalar(oldParamD1, oldRatio)
	scaledOldParamD2 := tensor.D2MulScalar(oldParamD2, oldRatio)
	scaledOldParamD3 := tensor.D3MulScalar(oldParamD3, oldRatio)

	err := model.perLayerD1Var.Param.Add(scaledOldParamD1)
	if err != nil {
		panic(err)
	}

	err = model.perLayerD2Var.Param.Add(scaledOldParamD2)
	if err != nil {
		panic(err)
	}

	return model.perLayerD3Var.Param.Add(scaledOldParamD3)
}

func (model *D1) UpdateGrad(x, t tensor.D1) error {
	_, _, bp, err := model.YAndLoss(x, t)
	if err != nil {
		return err
	}
	_, err = bp.Run()
	model.perLayerD1Var.AddL2RegularizationGrad(model.L2RegularizationD1Param)
	model.perLayerD2Var.AddL2RegularizationGrad(model.L2RegularizationD2Param)
	model.perLayerD3Var.AddL2RegularizationGrad(model.L2RegularizationD3Param)
	return err
}

func (model *D1) Train(lr float64) {
	model.perLayerD1Var.Train(lr)
	model.perLayerD2Var.Train(lr)
	model.perLayerD3Var.Train(lr)
}

func (model *D1) UpdateGradAndTrain(x, t tensor.D1, lr float64) {
	model.UpdateGrad(x, t)
	model.Train(lr)
}

func (model *D1) ValidateBackwardGrad(x, t tensor.D1) error {
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

	numGradD1 := mlfuncs.D1NumericalDifferentiation(model.perLayerD1Var.Param, lossD1)
	numGradD2 := mlfuncs.D2NumericalDifferentiation(model.perLayerD2Var.Param, lossD2)
	numGradD3 := mlfuncs.D3NumericalDifferentiation(model.perLayerD3Var.Param, lossD3)
	model.UpdateGrad(x, t)

	diffErrD1, err := tensor.D1Sub(model.perLayerD1Var.grad, numGradD1)
	if err != nil {
		return err
	}
	maxDiffErrD1 := diffErrD1.MapFunc(math.Abs).Max()

	diffErrD2, err := tensor.D2Sub(model.perLayerD2Var.grad, numGradD2)
	if err != nil {
		return err
	}
	maxDiffErrD2 := diffErrD2.MapFunc(math.Abs).Max().Max()

	diffErrD3, err := tensor.D3Sub(model.perLayerD3Var.grad, numGradD3)
	if err != nil {
		return err
	}
	maxDiffErrD3 := diffErrD3.MapFunc(math.Abs).Max().Max().Max()

	fmt.Println("maxDiffErrD1 =", maxDiffErrD1)
	fmt.Println("maxDiffErrD2 =", maxDiffErrD2)
	fmt.Println("maxDiffErrD3 =", maxDiffErrD3)
	return nil
}