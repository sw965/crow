package model

import (
	"github.com/sw965/crow/mlfuncs"
	"github.com/sw965/crow/optimizer"
	"github.com/sw965/crow/tensor"
)

type PerLayerD1Var struct {
	Param tensor.D1
	grad tensor.D1
	optimizer optimizer.D1Momentum
}

func (v *PerLayerD1Var) GetGrad() tensor.D1 {
	return v.grad
}

func (v *PerLayerD1Var) Init() {
	v.grad = make(tensor.D1, len(v.Param))
	velocity := make(tensor.D1, len(v.Param))
	v.optimizer = optimizer.NewD1Momentum(velocity)
}

func (v *PerLayerD1Var) L2Regularization(lambda float64) float64 {
	return mlfuncs.D1L2Regularization(v.Param, lambda)
}

func (v *PerLayerD1Var) AddL2RegularizationGrad(lambda float64) {
	l2Grad := mlfuncs.D1L2RegularizationDerivative(v.Param, lambda)
	v.grad.Add(l2Grad)
}

func (v *PerLayerD1Var) Train(lr float64) {
	v.optimizer.Train(v.Param, v.grad, lr)
}

type PerLayerD2Var struct {
	Param tensor.D2
	grad tensor.D2
	optimizer optimizer.D2Momentum
}

func (v *PerLayerD2Var) GetGrad() tensor.D2 {
	return v.grad
}

func (v *PerLayerD2Var) Init() {
	v.grad = tensor.NewD2ZerosLike(v.Param)
	velocity := tensor.NewD2ZerosLike(v.Param)
	v.optimizer = optimizer.NewD2Momentum(velocity)
}

func (v *PerLayerD2Var) L2Regularization(lambda float64) float64 {
	return mlfuncs.D2L2Regularization(v.Param, lambda)
}

func (v *PerLayerD2Var) AddL2RegularizationGrad(lambda float64) {
	l2Grad := mlfuncs.D2L2RegularizationDerivative(v.Param, lambda)
	v.grad.Add(l2Grad)
}

func (v *PerLayerD2Var) Train(lr float64) {
	v.optimizer.Train(v.Param, v.grad, lr)
}

type PerLayerD3Var struct {
	Param tensor.D3
	grad tensor.D3
	optimizer optimizer.D3Momentum
}

func (v *PerLayerD3Var) GetGrad() tensor.D3 {
	return v.grad
}

func (v *PerLayerD3Var) Init() {
	v.grad = tensor.NewD3ZerosLike(v.Param)
	velocity := tensor.NewD3ZerosLike(v.Param)
	v.optimizer = optimizer.NewD3Momentum(velocity)
}

func (v *PerLayerD3Var) L2Regularization(lambda float64) float64 {
	return mlfuncs.D3L2Regularization(v.Param, lambda)
}

func (v *PerLayerD3Var) AddL2RegularizationGrad(lambda float64) {
	l2Grad := mlfuncs.D3L2RegularizationDerivative(v.Param, lambda)
	v.grad.Add(l2Grad)
}

func (v *PerLayerD3Var) Train(lr float64) {
	v.optimizer.Train(v.Param, v.grad, lr)
}