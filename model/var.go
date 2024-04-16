package model

import (
	"github.com/sw965/crow/mlfuncs"
	"github.com/sw965/crow/optimizer"
	"github.com/sw965/crow/tensor"
)

type PerLayerScalarVar struct {
	Param tensor.D1
	Grad tensor.D1
	L2Lambda float64
	Optimizer optimizer.D1Momentum
}

func (v *PerLayerScalarVar) Init() {
	v.Grad = make(tensor.D1, len(v.Param))
	velocity := make(tensor.D1, len(v.Param))
	v.Optimizer = optimizer.D1Momentum{Velocity:velocity}
}

func (v *PerLayerScalarVar) L2Regularization() float64 {
	return mlfuncs.D1L2Regularization(v.L2Lambda)(v.Param)
}

func (v *PerLayerScalarVar) AddL2RegularizationGrad() {
	grad := mlfuncs.D1L2RegularizationDerivative(v.L2Lambda)(v.Param)
	v.Grad.Add(grad)
}

func (v *PerLayerScalarVar) Train(lr float64) {
	v.Optimizer.Train(v.Param, v.Grad, lr)
}

type PerLayerD1Var struct {
	Param tensor.D2
	Grad tensor.D2
	L2Lambda float64
	Optimizer optimizer.D2Momentum
}

func (v *PerLayerD1Var) Init() {
	v.Grad = tensor.NewD2ZerosLike(v.Param)
	velocity := tensor.NewD2ZerosLike(v.Param)
	v.Optimizer = optimizer.D2Momentum{Velocity:velocity}
}

func (v *PerLayerD1Var) L2Regularization() float64 {
	return mlfuncs.D2L2Regularization(v.L2Lambda)(v.Param)
}

func (v *PerLayerD1Var) AddL2RegularizationGrad() {
	grad := mlfuncs.D2L2RegularizationDerivative(v.L2Lambda)(v.Param)
	v.Grad.Add(grad)
}

func (v *PerLayerD1Var) Train(lr float64) {
	v.Optimizer.Train(v.Param, v.Grad, lr)
}

type PerLayerD2Var struct {
	Param tensor.D3
	Grad tensor.D3
	L2Lambda float64
	Optimizer optimizer.D3Momentum
}

func (v *PerLayerD2Var) Init() {
	v.Grad = tensor.NewD3ZerosLike(v.Param)
	velocity := tensor.NewD3ZerosLike(v.Param)
	v.Optimizer = optimizer.D3Momentum{Velocity:velocity}
}

func (v *PerLayerD2Var) L2Regularization() float64 {
	return mlfuncs.D3L2Regularization(v.L2Lambda)(v.Param)
}

func (v *PerLayerD2Var) AddL2RegularizationGrad() {
	grad := mlfuncs.D3L2RegularizationDerivative(v.L2Lambda)(v.Param)
	v.Grad.Add(grad)
}

func (v *PerLayerD2Var) Train(lr float64) {
	v.Optimizer.Train(v.Param, v.Grad, lr)
}