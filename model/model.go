package model

import (
	"fmt"
	"math"
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

	D1ParamLoss func(tensor.D1) float64
	D1ParamLossDerivative func(tensor.D1) tensor.D1

	D2ParamLoss func(tensor.D2) float64
	D2ParamLossDerivative func(tensor.D2) tensor.D2

	D3ParamLoss func(tensor.D3) float64
	D3ParamLossDerivative func(tensor.D3) tensor.D3

	L2NormGradClipThreshold float64
}

func NewD1(d1Var *D1Var, d2Var *D2Var, d3Var *D3Var) D1 {
	model := D1{d1Var:*d1Var, d2Var:*d2Var, d3Var:*d3Var}
	model.D1ParamLoss = func(_ tensor.D1) float64 { return 0.0 }
	model.D1ParamLossDerivative = func(w tensor.D1) tensor.D1 { return tensor.NewD1ZerosLike(w) }

	model.D2ParamLoss = func(_ tensor.D2) float64 { return 0.0 }
	model.D2ParamLossDerivative = func(w tensor.D2) tensor.D2 { return tensor.NewD2ZerosLike(w) }

	model.D3ParamLoss = func(_ tensor.D3) float64 { return 0.0 }
	model.D3ParamLossDerivative = func(w tensor.D3) tensor.D3 { return tensor.NewD3ZerosLike(w) }
	return model
}

func (model *D1) newBackPropagator(x, t tensor.D1) (*layer.D1BackPropagator, error) {
	y, backwards, err := model.Forwards.Run(x)
	if err != nil {
		return &layer.D1BackPropagator{}, err
	}
	_, lossBackward, err := model.LossForward(y, t)
	bp := layer.NewD1BackPropagator(backwards, lossBackward)
	return &bp, err
}

func (model *D1) Predict(x tensor.D1) (tensor.D1, error) {
	y, _, err := model.Forwards.Run(x)
	return y, err
}

func (model *D1) MeanLoss(x, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("入力値と正解ラベルのバッチ数が一致しないため、平均損失が計算できません。一致するようにしてください。")
	}

	paramLoss := model.D1ParamLoss(model.d1Var.Param) + model.D2ParamLoss(model.d2Var.Param) + model.D3ParamLoss(model.d3Var.Param)
	sum := paramLoss * float64(n)

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
	}
	mean := sum / float64(n)
	return mean, nil
}

func (model *D1) Accuracy(x, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("入力と正解ラベルのバッチ数が一致しません。Accuracyを求めるには、バッチ数が一致している必要があります。")
	}
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

	gradD1 := model.D1ParamLossDerivative(model.d1Var.Param)
	gradD2 := model.D2ParamLossDerivative(model.d2Var.Param)
	gradD3 := model.D3ParamLossDerivative(model.d3Var.Param)

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

func (model *D1) Train(x, t tensor.D1, lr float64) {
	model.UpdateGrad(x, t)
	model.d1Var.Train(lr)
	model.d2Var.Train(lr)
	model.d3Var.Train(lr)
}

func (model *D1) ValidateBackwardAndNumericalGradientDifference(x, t tensor.D1) error {
	lossD1 := func(_ tensor.D1) float64 {
		loss, err := model.MeanLoss(tensor.D2{x}, tensor.D2{t})
		if err != nil {
			panic(err)
		}
		return loss
	}

	lossD2 := func(_ tensor.D2) float64 {
		loss, err := model.MeanLoss(tensor.D2{x}, tensor.D2{t})
		if err != nil {
			panic(err)
		}
		return loss
	}

	lossD3 := func(d2Params tensor.D3) float64 {
		loss, err := model.MeanLoss(tensor.D2{x}, tensor.D2{t})
		if err != nil {
			panic(err)
		}
		return loss
	} 

	numGradD1 := mlfuncs.D1NumericalDifferentiation(model.d1Var.Param, lossD1)
	numGradD2 := mlfuncs.D2NumericalDifferentiation(model.d2Var.Param, lossD2)
	numGradD3 := mlfuncs.D3NumericalDifferentiation(model.d3Var.Param, lossD3)
	mlfuncs.ClipL2Norm(numGradD1, numGradD2, numGradD3, model.L2NormGradClipThreshold)
	model.UpdateGrad(x, t)

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

type D2LinearSum struct {
	w D2Var
	b D1Var
	forward layer.D2LinearSumForward
	OutputForward layer.D1Forward
	LossForward layer.D1LossForward
	l2Regularization float64
}

func NewD2LinearSum(w D2Var, b D1Var, lambda float64) D2LinearSum {
	forward := layer.NewD2LinearSumForward(w.Param, b.Param, w.GetGrad(), b.GetGrad())
	return D2LinearSum{w:w, b:b, forward:forward, l2Regularization:lambda}
}

func NewD2LinearSumTanhMSE(w D2Var, b D1Var, lambda float64) D2LinearSum {
	model := NewD2LinearSum(w, b, lambda)
	model.OutputForward = layer.NewD1TanhForward()
	model.LossForward = layer.NewD1MeanSquaredErrorForward()
	return model
}

func (model *D2LinearSum) Predict(x tensor.D2) (tensor.D1, error) {
	u1, _, err := model.forward(x)
	if err != nil {
		return tensor.D1{}, err
	}
	y, _, err := model.OutputForward(u1, make(layer.D1Backwards, 0, 1))
	return y, err
}

func (model *D2LinearSum) MeanLoss(x tensor.D3, t tensor.D2) (float64, error) {
	n := len(x)
	if len(x) != len(t) {
		return 0.0, fmt.Errorf("入力と正解ラベルのバッチ数が一致しません。平均誤差を求めるには、バッチ数が一致する必要があります。")
	}

	sum := mlfuncs.D2L2Regularization(model.w.Param, model.l2Regularization) * float64(n)
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
	}
	mean := sum / float64(n)
	return mean, nil
}

func (model *D2LinearSum) Accuracy(x tensor.D3, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("入力と正解ラベルのバッチ数が一致しません。Accuracyを求めるには、バッチ数が一致している必要があります。")
	}
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

func (model *D2LinearSum) UpdateGrad(x tensor.D2, t tensor.D1) error {
	u1, backward, err := model.forward(x)
	if err != nil {
		return err
	}

	y, yBackwards, err := model.OutputForward(u1, make(layer.D1Backwards, 0, 1))
	if err != nil {
		return err
	}
	yBackward := yBackwards[0]

	_, lossBackward, err := model.LossForward(y, t)
	if err != nil {
		return err
	}

	chain, err := lossBackward()
	if err != nil {
		return err
	}

	chain, err = yBackward(chain)
	if err != nil {
		return err
	}
	_, err = backward(chain)
	return err
}

func (model *D2LinearSum) Train(lr float64) {
	model.w.Train(lr)
	model.b.Train(lr)
}

func (model *D2LinearSum) ValidateBackwardAndNumericalGradientDifference(x tensor.D2, t tensor.D1) error {
	lossW := func(w tensor.D2) float64 {
		loss, err := model.MeanLoss(tensor.D3{x}, tensor.D2{t})
		if err != nil {
			panic(err)
		}
		return loss
	}

	lossB := func(b tensor.D1) float64 {
		loss, err := model.MeanLoss(tensor.D3{x}, tensor.D2{t})
		if err != nil {
			panic(err)
		}
		return loss
	}

	numGradW := mlfuncs.D2NumericalDifferentiation(model.w.Param, lossW)
	numGradB := mlfuncs.D1NumericalDifferentiation(model.b.Param, lossB)

	model.UpdateGrad(x, t)
	gradW := model.w.GetGrad()
	gradB := model.b.GetGrad()

	diffErrW, err := tensor.D2Sub(numGradW, gradW)
	if err != nil {
		return err
	}
	diffErrB, err := tensor.D1Sub(numGradB, gradB)
	if err != nil {
		return err
	}

	maxDiffErrW := diffErrW.MapFunc(math.Abs).MaxAxisRow().Max()
	maxDiffErrB := diffErrB.MapFunc(math.Abs).Max()
	fmt.Println("maxDiffErrW =", maxDiffErrW)
	fmt.Println("maxDiffErrB =", maxDiffErrB)
	return nil
}