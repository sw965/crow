package model

import (
	"fmt"
	"math"
	"github.com/sw965/crow/layer"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
	"github.com/sw965/crow/optimizer"
)

type D1Var struct {
	Param tensor.D1
	grad tensor.D1
	optimizer optimizer.D1Momentum
}

func (v *D1Var) GetGrad() tensor.D1 {
	return v.grad
}

func (v *D1Var) Init(momentum float64) {
	v.grad = make(tensor.D1, len(v.Param))
	velocity := make(tensor.D1, len(v.Param))
	v.optimizer = optimizer.NewD1Momentum(momentum, velocity)
}

func (v *D1Var) Train(lr float64) {
	v.optimizer.Train(v.Param, v.grad, lr)
}

type D2Var struct {
	Param tensor.D2
	grad tensor.D2
	optimizer optimizer.D2Momentum
}

func (v *D2Var) GetGrad() tensor.D2 {
	return v.grad
}

func (v *D2Var) Init(momentum float64) {
	v.grad = tensor.NewD2ZerosLike(v.Param)
	velocity := tensor.NewD2ZerosLike(v.Param)
	v.optimizer = optimizer.NewD2Momentum(momentum, velocity)
}

func (v *D2Var) Train(lr float64) {
	v.optimizer.Train(v.Param, v.grad, lr)
}

type D3Var struct {
	Param tensor.D3
	grad tensor.D3
	optimizer optimizer.D3Momentum
}

func (v *D3Var) GetGrad() tensor.D3 {
	return v.grad
}

func (v *D3Var) Init(momentum float64) {
	v.grad = tensor.NewD3ZerosLike(v.Param)
	velocity := tensor.NewD3ZerosLike(v.Param)
	v.optimizer = optimizer.NewD3Momentum(momentum, velocity)
}

func (v *D3Var) Train(lr float64) {
	v.optimizer.Train(v.Param, v.grad, lr)
}

type D1 struct {
	d1Var D1Var
	d2Var D2Var
	d3Var D3Var

	Forwards layer.D1Forwards
	LossForward layer.D1LossForward

	ParamLoss func(tensor.D1, tensor.D2, tensor.D3) float64
	ParamLossDerivative func(tensor.D1, tensor.D2, tensor.D3) (tensor.D1, tensor.D2, tensor.D3)
}

func NewD1(d1Var *D1Var, d2Var *D2Var, d3Var *D3Var) D1 {
	return D1{d1Var:*d1Var, d2Var:*d2Var, d3Var:*d3Var}
}

func (model *D1) Predict(x tensor.D1) (tensor.D1, layer.D1Backwards, error) {
	y, backwards, err := model.Forwards.Run(x)
	return y, backwards, err
}

func (model *D1) YAndLoss(x, t tensor.D1) (tensor.D1, float64, *layer.D1BackPropagator, error) {
	y, backwards, err := model.Predict(x)
	loss, lossBackward, err := model.LossForward(y, t)
	loss += model.ParamLoss(model.d1Var.Param, model.d2Var.Param, model.d3Var.Param)
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

func (model *D1) SWA(scale float64, oldParamD1 tensor.D1, oldParamD2 tensor.D2, oldParamD3 tensor.D3) error {
	model.d1Var.Param.MulScalar(scale)
	model.d2Var.Param.MulScalar(scale)
	model.d3Var.Param.MulScalar(scale)

	oldScale := 1.0 - scale
	scaledOldParamD1 := tensor.D1MulScalar(oldParamD1, oldScale)
	scaledOldParamD2 := tensor.D2MulScalar(oldParamD2, oldScale)
	scaledOldParamD3 := tensor.D3MulScalar(oldParamD3, oldScale)

	err := model.d1Var.Param.Add(scaledOldParamD1)
	if err != nil {
		panic(err)
	}

	err = model.d2Var.Param.Add(scaledOldParamD2)
	if err != nil {
		panic(err)
	}

	return model.d3Var.Param.Add(scaledOldParamD3)
}

func (model *D1) UpdateGrad(x, t tensor.D1, threshold float64) error {
	_, _, bp, err := model.YAndLoss(x, t)
	if err != nil {
		return err
	}
	_, err = bp.Run()
	gradD1, gradD2, gradD3 := model.ParamLossDerivative(
		model.d1Var.Param, model.d2Var.Param, model.d3Var.Param,
	)

	err = model.d1Var.grad.Add(gradD1)
	if err != nil {
		return err
	}

	err = model.d2Var.grad.Add(gradD2)
	if err != nil {
		return err
	}

	err = model.d3Var.grad.Add(gradD3)
	if err != nil {
		return err
	}

	if threshold > 0.0 {
		mlfuncs.ClipL2Norm(model.d1Var.grad, model.d2Var.grad, model.d3Var.grad, threshold)
	}
	return err
}

func (model *D1) Train(lr float64) {
	model.d1Var.Train(lr)
	model.d2Var.Train(lr)
	model.d3Var.Train(lr)
}

func (model *D1) UpdateGradAndTrain(x, t tensor.D1, lr, threshold float64) {
	model.UpdateGrad(x, t, threshold)
	model.Train(lr)
}

func (model *D1) ValidateBackwardGrad(x, t tensor.D1, threshold float64) error {
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

	numGradD1 := mlfuncs.D1NumericalDifferentiation(model.d1Var.Param, lossD1)
	numGradD2 := mlfuncs.D2NumericalDifferentiation(model.d2Var.Param, lossD2)
	numGradD3 := mlfuncs.D3NumericalDifferentiation(model.d3Var.Param, lossD3)
	mlfuncs.ClipL2Norm(numGradD1, numGradD2, numGradD3, threshold)
	model.UpdateGrad(x, t, threshold)

	diffErrD1, err := tensor.D1Sub(model.d1Var.grad, numGradD1)
	if err != nil {
		return err
	}
	maxDiffErrD1 := diffErrD1.MapFunc(math.Abs).Max()

	diffErrD2, err := tensor.D2Sub(model.d2Var.grad, numGradD2)
	if err != nil {
		return err
	}
	maxDiffErrD2 := diffErrD2.MapFunc(math.Abs).MaxAxisRow().Max()

	diffErrD3, err := tensor.D3Sub(model.d3Var.grad, numGradD3)
	if err != nil {
		return err
	}
	maxDiffErrD3 := diffErrD3.MapFunc(math.Abs).MaxAxisRow().MaxAxisRow().Max()

	fmt.Println("maxDiffErrD1 =", maxDiffErrD1)
	fmt.Println("maxDiffErrD2 =", maxDiffErrD2)
	fmt.Println("maxDiffErrD3 =", maxDiffErrD3)
	return nil
}