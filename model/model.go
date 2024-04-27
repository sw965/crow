package model

import (
	"fmt"
	"github.com/sw965/crow/layer"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
	"github.com/sw965/crow/optimizer"
	"github.com/sw965/omw"
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

	L2NormGradClipThreshold float64
}

func NewD1(d1Var *D1Var, d2Var *D2Var, d3Var *D3Var) D1 {
	return D1{d1Var:*d1Var, d2Var:*d2Var, d3Var:*d3Var}
}

func (model *D1) predict(x tensor.D1) (tensor.D1, layer.D1Backwards, error) {
	y, backwards, err := model.Forwards.Run(x)
	return y, backwards, err
}

func (model *D1) Predict(x tensor.D1) (tensor.D1, error) {
	y, _, err := model.Forwards.Run(x)
	return y, err
}

func (model *D1) newBackPropagator(x, t tensor.D1) (*layer.D1BackPropagator, error) {
	y, backwards, err := model.predict(x)
	if err != nil {
		return &layer.D1BackPropagator{}, err
	}
	_, lossBackward, err := model.LossForward(y, t)
	bp := layer.NewD1BackPropagator(backwards, lossBackward)
	return &bp, err
}

func (model *D1) MeanLoss(x, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("入力値と正解ラベルのバッチ数が一致しないため、平均損失が計算できません。一致するようにしてください。")
	}

	sum := 0.0
	for i := range x {
		y, err := model.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		loss, _, err := model.LossForward(y, t[i])
		if err != nil {
			return 0.0, err
		}
		sum += loss
		sum += model.ParamLoss(model.d1Var.Param, model.d2Var.Param, model.d3Var.Param)
	}

	mean := sum / float64(n)
	return mean, nil
}

func (model *D1) Accuracy(x, t tensor.D2) (float64, error) {
	n := len(x)
	correct := 0
	for i := range x {
		y, err := model.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		if omw.MaxIndex(y) == omw.MaxIndex(t[i]) {
			correct += 1
		}
	}
	return float64(correct) / float64(n), nil
}

func (model *D1) UpdateGrad(x, t tensor.D1) error {
	bp, err := model.newBackPropagator(x, t)
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

	if model.L2NormGradClipThreshold > 0.0 {
		mlfuncs.ClipL2Norm(model.d1Var.grad, model.d2Var.grad, model.d3Var.grad, model.L2NormGradClipThreshold)
	}
	return err
}

func (model *D1) Train(lr float64) {
	model.d1Var.Train(lr)
	model.d2Var.Train(lr)
	model.d3Var.Train(lr)
}

func (model *D1) UpdateGradAndTrain(x, t tensor.D1, lr float64) {
	model.UpdateGrad(x, t)
	model.Train(lr)
}