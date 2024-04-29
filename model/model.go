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

func (model *D1) Predict(x tensor.D1) (tensor.D1, error) {
	y, _, err := model.Forwards.Run(x)
	return y, err
}

func (model *D1) MeanLoss(x, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("入力値と正解ラベルのバッチ数が一致しないため、平均損失が計算できません。一致するようにしてください。")
	}

	sum := model.D1ParamLoss(model.d1Var.Param)
	sum += model.D2ParamLoss(model.d2Var.Param)
	sum += model.D3ParamLoss(model.d3Var.Param)
	sum *= float64(n)

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

func (model *D1) newBackPropagator(x, t tensor.D1) (*layer.D1BackPropagator, error) {
	y, backwards, err := model.Forwards.Run(x)
	if err != nil {
		return &layer.D1BackPropagator{}, err
	}
	_, lossBackward, err := model.LossForward(y, t)
	bp := layer.NewD1BackPropagator(backwards, lossBackward)
	return &bp, err
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
	w tensor.D2
	b tensor.D1
	outputFunc func(tensor.D1) tensor.D1
	outputDerivative func(tensor.D1) tensor.D1
	lossFunc func(tensor.D1, tensor.D1) (float64, error)
	lossDerivative func(tensor.D1, tensor.D1) (tensor.D1, error)
	l2Regularization float64
}

func NewD2LinearSum(
	output,
	outputPrime func(tensor.D1) tensor.D1,
	loss func(tensor.D1, tensor.D1) (float64, error),
	lossPrime func(tensor.D1, tensor.D1) (tensor.D1, error),
	l2 float64) D2LinearSum {

	return D2LinearSum{
		outputFunc:output,
		outputDerivative:outputPrime,
		lossFunc:loss,
		lossDerivative:lossPrime,
		l2Regularization:l2,
	}
}

func NewD2LinearSumTanhMSE(l2 float64) D2LinearSum {
	return NewD2LinearSum(
		mlfuncs.D1Tanh,
		mlfuncs.D1TanhDerivative,
		mlfuncs.D1MeanSquaredError,
		mlfuncs.D1MeanSquaredErrorDerivative,
		l2,
	)
}

func (model *D2LinearSum) SetParam(w tensor.D2, b tensor.D1) {
	model.w = w
	model.b = b
}

func (model *D2LinearSum) Predict(x tensor.D2) (tensor.D1, error) {
	u, err := mlfuncs.D2LinearSum(x, model.w, model.b)
	y := model.outputFunc(u)
	return y, err
}

func (model *D2LinearSum) Grad(x tensor.D2, t tensor.D1) (tensor.D2, tensor.D2, tensor.D1, error) {
	y, err := model.Predict(x)
	if err != nil {
		return tensor.D2{}, tensor.D2{}, tensor.D1{}, err
	}

	dLdy, err := model.lossDerivative(y, t)
	if err != nil {
		return tensor.D2{}, tensor.D2{}, tensor.D1{}, err
	}

	dydx, dydw, _, err := mlfuncs.D2LinearSumDerivative(x, model.w)
	if err != nil {
		return tensor.D2{}, tensor.D2{}, tensor.D1{}, err
	}

	dLdx :=  tensor.NewD2ZerosLike(dydx)
	for i := range dLdx {
		for j := range dLdx[i] {
			dLdx[i][j] = dydx[i][j] * dLdy[i]
		}
	}

	dLdw := tensor.NewD2ZerosLike(dydw)
	for i := range dLdw {
		for j := range dLdw[i] {
			dLdw[i][j] = dydw[i][j] * dLdy[i]
		}
	}
	dLdb := dLdy
	return dLdx, dLdw, dLdb, nil
}

func (model *D2LinearSum) Train(x tensor.D2, t tensor.D1, lr float64) error {
	_, dLdw, dLdb, err := model.Grad(x, t)
	dLdw.MulScalar(lr)
	dLdb.MulScalar(lr)
	model.w.Sub(dLdw)
	model.b.Sub(dLdb)
	return err
}

func (model *D2LinearSum) MeanLoss(x tensor.D3, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("入力と正解ラベルのバッチ数が一致しません。平均誤差を求めるには、バッチ数が一致する必要があります。")
	}

	sum := mlfuncs.D2L2Regularization(model.w, model.l2Regularization) * float64(n)
	for i := range x {
		y, err := model.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		loss, err := model.lossFunc(y, t[i])
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

func (model *D2LinearSum) ValidateBackwardAndNumericalGradientDifference(x tensor.D2, t tensor.D1) error {
	lossX := func(x tensor.D2) float64 {
		loss, err := model.MeanLoss(tensor.D3{x}, tensor.D2{t})
		if err != nil {
			panic(err)
		}
		return loss
	}

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

	numGradX := mlfuncs.D2NumericalDifferentiation(x, lossX)
	numGradW := mlfuncs.D2NumericalDifferentiation(model.w, lossW)
	numGradB := mlfuncs.D1NumericalDifferentiation(model.b, lossB)

	gradX, gradW, gradB, err := model.Grad(x, t)
	if err != nil {
		return err
	}

	diffErrX, err := tensor.D2Sub(numGradX, gradX)
	if err != nil {
		return err
	}
	diffErrW, err := tensor.D2Sub(numGradW, gradW)
	if err != nil {
		return err
	}
	diffErrB, err := tensor.D1Sub(numGradB, gradB)
	if err != nil {
		return err
	}

	maxDiffErrX := diffErrX.MapFunc(math.Abs).MaxAxisRow().Max()
	maxDiffErrW := diffErrW.MapFunc(math.Abs).MaxAxisRow().Max()
	maxDiffErrB := diffErrB.MapFunc(math.Abs).Max()
	fmt.Println("maxDiffErrX =", maxDiffErrX)
	fmt.Println("maxDiffErrW =", maxDiffErrW)
	fmt.Println("maxDiffErrB =", maxDiffErrB)
	return nil
}