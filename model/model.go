package model

import (
	"fmt"
	"math"
	"github.com/sw965/crow/layer/1d"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
	"github.com/sw965/crow/mlfuncs/1d"
	"github.com/sw965/crow/mlfuncs/2d"
	"github.com/sw965/crow/mlfuncs/3d"
	"github.com/sw965/omw"
)

type Variable struct {
	Param1D tensor.D1
	Param2D tensor.D2
	Param3D tensor.D3

	grad1D tensor.D1
	grad2D tensor.D2
	grad3D tensor.D3
}

func (v *Variable) GetGrad1D() tensor.D1 {
	return v.grad1D
}

func (v *Variable) GetGrad2D() tensor.D2 {
	return v.grad2D
}

func (v *Variable) GetGrad3D() tensor.D3 {
	return v.grad3D
}

func (v *Variable) Init() {
	v.grad1D = tensor.NewD1ZerosLike(v.Param1D)
	v.grad2D = tensor.NewD2ZerosLike(v.Param2D)
	v.grad3D = tensor.NewD3ZerosLike(v.Param3D)
}

func (v *Variable) GradMulScaler(lr float64) {
	v.grad1D.MulScalar(lr)
	v.grad2D.MulScalar(lr)
	v.grad3D.MulScalar(lr)	
}

func (v *Variable) SGD() {
	v.Param1D.Sub(v.grad1D)
	v.Param2D.Sub(v.grad2D)
	v.Param3D.Sub(v.grad3D)
}

type SequentialInputOutput1D struct {
	variable Variable

	Forwards layer1d.Forwards

	YLossFunc func(tensor.D1, tensor.D1) (float64, error)
	YLossDerivative func(tensor.D1, tensor.D1) (tensor.D1, error)

	YLossForward layer1d.YLossForward

	Param1DLossFunc func(tensor.D1)float64
	Param2DLossFunc func(tensor.D2)float64
	Param3DLossFunc func(tensor.D3)float64

	Param1DLossDerivative func(tensor.D1)tensor.D1
	Param2DLossDerivative func(tensor.D2)tensor.D2
	Param3DLossDerivative func(tensor.D3)tensor.D3

	L2NormGradClipThreshold float64
}

func NewSequentialInputOutput1D(variable Variable) SequentialInputOutput1D {
	return SequentialInputOutput1D{
		variable:variable,
		Param1DLossFunc:func(_ tensor.D1) float64 { return 0.0 },
		Param1DLossDerivative:tensor.NewD1ZerosLike,
		Param2DLossFunc:func(_ tensor.D2) float64 { return 0.0 },
		Param2DLossDerivative:tensor.NewD2ZerosLike,
		Param3DLossFunc:func(_ tensor.D3) float64 { return 0.0 },
		Param3DLossDerivative:tensor.NewD3ZerosLike,
	}
}

func (m *SequentialInputOutput1D) Predict(x tensor.D1) (tensor.D1, error) {
	y, _, err := m.Forwards.Propagate(x)
	return y, err
}

func (m *SequentialInputOutput1D) MeanLoss(x, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("入力値と正解ラベルのバッチ数が一致しません。")
	}

	sum := m.Param1DLossFunc(m.variable.Param1D)
	sum += m.Param2DLossFunc(m.variable.Param2D)
	sum += m.Param3DLossFunc(m.variable.Param3D)
	sum *= float64(n)

	for i := range x {
		y, err := m.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		yLoss, err := m.YLossFunc(y, t[i])
		if err != nil {
			return 0.0, err
		}
		sum += yLoss
	}
	mean := sum / float64(n)
	return mean, nil
}

func (m *SequentialInputOutput1D) Accuracy(x, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("入力と正解ラベルのバッチ数が一致しません。")
	}
	correct := 0
	for i := range x {
		y, err := m.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		if omw.MaxIndex(y) == omw.MaxIndex(t[i]) {
			correct += 1
		}
	}
	return float64(correct) / float64(n), nil
}

func (m *SequentialInputOutput1D) UpdateGrad(x, t tensor.D1) error {
	y, backwards, err := m.Forwards.Propagate(x)
	if err != nil {
		return err
	}

	dLdy, err := m.YLossDerivative(y, t)
	if err != nil {
		return err
	}

	_, err = backwards.Propagate(dLdy)
	if err != nil {
		return err
	}

	grad1D := m.Param1DLossDerivative(m.variable.Param1D)
	grad2D := m.Param2DLossDerivative(m.variable.Param2D)
	grad3D := m.Param3DLossDerivative(m.variable.Param3D)

	err = m.variable.grad1D.Add(grad1D)
	if err != nil {
		return err
	}

	err = m.variable.grad2D.Add(grad2D)
	if err != nil {
		return err
	}

	err = m.variable.grad3D.Add(grad3D)
	if err != nil {
		return err
	}

	if m.L2NormGradClipThreshold > 0.0 {
		mlfuncs.ClipL2Norm(m.variable.grad1D, m.variable.grad2D, m.variable.grad3D, m.L2NormGradClipThreshold)
	}
	return err
}

func (m *SequentialInputOutput1D) SGD(x, t tensor.D1, lr float64) {
	m.UpdateGrad(x, t)
	m.variable.GradMulScaler(lr)
	m.variable.SGD()
}

func (m *SequentialInputOutput1D) ValidateBackwardAndNumericalGradientDifference(x, t tensor.D1) error {
	lossD1 := func(_ tensor.D1) float64 {
		loss, err := m.MeanLoss(tensor.D2{x}, tensor.D2{t})
		if err != nil {
			panic(err)
		}
		return loss
	}

	lossD2 := func(_ tensor.D2) float64 {
		loss, err := m.MeanLoss(tensor.D2{x}, tensor.D2{t})
		if err != nil {
			panic(err)
		}
		return loss
	}

	lossD3 := func(d2Params tensor.D3) float64 {
		loss, err := m.MeanLoss(tensor.D2{x}, tensor.D2{t})
		if err != nil {
			panic(err)
		}
		return loss
	} 

	numGradD1 := mlfuncs1d.NumericalDifferentiation(m.variable.Param1D, lossD1)
	numGradD2 := mlfuncs2d.NumericalDifferentiation(m.variable.Param2D, lossD2)
	numGradD3 := mlfuncs3d.NumericalDifferentiation(m.variable.Param3D, lossD3)
	mlfuncs.ClipL2Norm(numGradD1, numGradD2, numGradD3, m.L2NormGradClipThreshold)
	m.UpdateGrad(x, t)

	diffD1, err := tensor.D1Sub(m.variable.grad1D, numGradD1)
	if err != nil {
		return err
	}
	maxDiffD1 := diffD1.MapFunc(math.Abs).Max()

	diffD2, err := tensor.D2Sub(m.variable.grad2D, numGradD2)
	if err != nil {
		return err
	}
	maxDiffD2 := diffD2.MapFunc(math.Abs).MaxRow().Max()

	diffD3, err := tensor.D3Sub(m.variable.grad3D, numGradD3)
	if err != nil {
		return err
	}
	maxDiffD3 := diffD3.MapFunc(math.Abs).MaxRow().MaxRow().Max()

	fmt.Println("maxDiffD1 =", maxDiffD1)
	fmt.Println("maxDiffD2 =", maxDiffD2)
	fmt.Println("maxDiffD3 =", maxDiffD3)
	return nil
}

type LinearSum struct {
	W tensor.D2
	B tensor.D1
	YFunc func(tensor.D1) tensor.D1
	YDerivative func(tensor.D1) tensor.D1
	YLossFunc func(tensor.D1, tensor.D1) (float64, error)
	YLossDerivative func(tensor.D1, tensor.D1) (tensor.D1, error)
	WLossFunc func(tensor.D2) float64
	WLossDerivative func(tensor.D2) tensor.D2
}

func NewLinearSumIdentityMSE(c float64) LinearSum {
	return LinearSum{
		YFunc:mlfuncs.Identity[tensor.D1],
		YDerivative:tensor.NewD1OnesLike,
		YLossFunc:mlfuncs1d.MeanSquaredError,
		YLossDerivative:mlfuncs1d.MeanSquaredErrorDerivative,
		WLossFunc:mlfuncs2d.L2Regularization(c),
		WLossDerivative:mlfuncs2d.L2RegularizationDerivative(c),
	}
}

func NewLinearSumSigmoidMSE(c float64) LinearSum {
	return LinearSum{
		YFunc:mlfuncs1d.Sigmoid,
		YDerivative:mlfuncs1d.SigmoidDerivative,
		YLossFunc:mlfuncs1d.MeanSquaredError,
		YLossDerivative:mlfuncs1d.MeanSquaredErrorDerivative,
		WLossFunc:mlfuncs2d.L2Regularization(c),
		WLossDerivative:mlfuncs2d.L2RegularizationDerivative(c),
	}
}

func (m *LinearSum) Predict(x tensor.D2) (tensor.D1, error) {
	u, err := mlfuncs2d.LinearSum(x, m.W, m.B)
	y := m.YFunc(u)
	return y, err
}

func (m *LinearSum) Grad(x tensor.D2, t tensor.D1) (tensor.D2, tensor.D2, tensor.D1, error) {
	u, err := mlfuncs2d.LinearSum(x, m.W, m.B)
	if err != nil {
		return tensor.D2{}, tensor.D2{}, tensor.D1{}, err
	}

	y, err := m.Predict(x)
	if err != nil {
		return tensor.D2{}, tensor.D2{}, tensor.D1{}, err
	}

	//ここから局所的な微分
	dLdy, err := m.YLossDerivative(y, t)
	if err != nil {
		return tensor.D2{}, tensor.D2{}, tensor.D1{}, err
	}

	dydu := m.YDerivative(u)

	dudx, dudw, _, err := mlfuncs2d.LinearSumDerivative(x, m.W)
	if err != nil {
		return tensor.D2{}, tensor.D2{}, tensor.D1{}, err
	}
	//ここまで局所的な微分

	//ここから連鎖律 (損失Lをx, w, b について微分)
	dLdu, err := tensor.D1Mul(dydu, dLdy)
	if err != nil {
		return tensor.D2{}, tensor.D2{}, tensor.D1{}, err
	}

	// ∂L/∂x
	dLdx, err := tensor.D2MulD1Col(dudx, dLdu)
	if err != nil {
		return tensor.D2{}, tensor.D2{}, tensor.D1{}, err
	}

	// ∂L/∂w
	dLdw, err := tensor.D2MulD1Col(dudw, dLdu)
	if err != nil {
		return tensor.D2{}, tensor.D2{}, tensor.D1{}, err
	}
	err = dLdw.Add(m.WLossDerivative(m.W))

	// ∂L/∂b
	dLdb := dLdu
	return dLdx, dLdw, dLdb, err
}

func (m *LinearSum) SGD(x tensor.D2, t tensor.D1, lr float64) error {
	_, dLdw, dLdb, err := m.Grad(x, t)
	dLdw.MulScalar(lr)
	dLdb.MulScalar(lr)
	m.W.Sub(dLdw)
	m.B.Sub(dLdb)
	return err
}

func (m *LinearSum) MeanLoss(x tensor.D3, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("入力と正解ラベルのバッチ数が一致しません。")
	}

	sum := m.WLossFunc(m.W) * float64(n)

	for i := range x {
		y, err := m.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		yLoss, err := m.YLossFunc(y, t[i])
		if err != nil {
			return 0.0, err
		}
		sum += yLoss
	}
	mean := sum / float64(n)
	return mean, nil
}

func (m *LinearSum) Accuracy(x tensor.D3, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("入力と正解ラベルのバッチ数が一致しません。")
	}
	correct := 0
	for i := range x {
		y, err := m.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		if omw.MaxIndex(y) == omw.MaxIndex(t[i]) {
			correct += 1
		}
	}
	return float64(correct) / float64(n), nil
}

func (m *LinearSum) ValidateBackwardAndNumericalGradientDifference(x tensor.D2, t tensor.D1) error {
	lossFunc := func(x, w tensor.D2, b tensor.D1) float64 {
		loss, err := m.MeanLoss(tensor.D3{x}, tensor.D2{t})
		if err != nil {
			panic(err)
		}
		return loss
	}

	xLossFunc := func(x tensor.D2) float64 { return lossFunc(x, m.W, m.B) }
	wLossFunc := func(w tensor.D2) float64 { return lossFunc(x, w, m.B) }
	bLossFunc := func(b tensor.D1) float64 { return lossFunc(x, m.W, b) }

	numGradX := mlfuncs2d.NumericalDifferentiation(x, xLossFunc)
	numGradW := mlfuncs2d.NumericalDifferentiation(m.W, wLossFunc)
	numGradB := mlfuncs1d.NumericalDifferentiation(m.B, bLossFunc)

	gradX, gradW, gradB, err := m.Grad(x, t)
	if err != nil {
		return err
	}

	diffX, err := tensor.D2Sub(numGradX, gradX)
	if err != nil {
		return err
	}
	diffW, err := tensor.D2Sub(numGradW, gradW)
	if err != nil {
		return err
	}
	
	diffB, err := tensor.D1Sub(numGradB, gradB)
	if err != nil {
		return err
	}

	maxDiffX := diffX.MapFunc(math.Abs).MaxRow().Max()
	maxDiffW := diffW.MapFunc(math.Abs).MaxRow().Max()
	maxDiffB := diffB.MapFunc(math.Abs).Max()
	fmt.Println("maxDiffX =", maxDiffX)
	fmt.Println("maxDiffW =", maxDiffW)
	fmt.Println("maxDiffB =", maxDiffB)
	return nil
}