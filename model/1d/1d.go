package model1d

import (
	"fmt"
	"github.com/sw965/crow/layer/1d"
	"github.com/sw965/crow/ml/1d"
	"github.com/sw965/crow/ml/2d"
	"github.com/sw965/crow/ml/3d"
	"github.com/sw965/crow/tensor"
	omwjson "github.com/sw965/omw/json"
	omwslices "github.com/sw965/omw/slices"
	"math"
	"math/rand"
)

type Param struct {
	D1 tensor.D1
	D2 tensor.D2
	D3 tensor.D3
}

func LoadParamJSON(path string) (Param, error) {
	param, err := omwjson.Load[Param](path)
	return param, err
}

func (p *Param) WriteJSON(path string) error {
	err := omwjson.Write[Param](p, path)
	return err
}

type Variable struct {
	Param Param
	grad1D tensor.D1
	grad2D tensor.D2
	grad3D tensor.D3
	initMethodCallCount int
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

func (v *Variable) Init() error {
	if v.initMethodCallCount == 1 {
		return fmt.Errorf("Variable.Initを呼び出すのは2度目です。1度しか呼び出す事は出来ません。")
	}
	//ここの部分でアドレスが書き換わる為、2回呼び出すと勾配の更新が出来なくなる。
	v.grad1D = tensor.NewD1ZerosLike(v.Param.D1)
	v.grad2D = tensor.NewD2ZerosLike(v.Param.D2)
	v.grad3D = tensor.NewD3ZerosLike(v.Param.D3)
	v.initMethodCallCount += 1
	return nil
}

func (v *Variable) ResetGrad() {
	v.grad1D.Copy(tensor.NewD1ZerosLike(v.grad1D))
	v.grad2D.Copy(tensor.NewD2ZerosLike(v.grad2D))
	v.grad3D.Copy(tensor.NewD3ZerosLike(v.grad3D))
}

func (v *Variable) ComputeGradL2Norm() float64 {
	sqSum := 0.0
	for _, e := range v.grad1D {
		sqSum += (e * e)
	}

	for _, ei := range v.grad2D {
		for _, eij := range ei {
			sqSum += (eij * eij)
		}
	}

	for _, ei := range v.grad3D {
		for _, eij := range ei {
			for _, eijk := range eij {
				sqSum += (eijk * eijk)
			}
		}
	}
	return math.Sqrt(sqSum)
}

func (v *Variable) ClipGrads(maxNorm float64) {
	norm := v.ComputeGradL2Norm()
	scale := maxNorm / norm
	if scale < 1.0 {
		v.grad1D.MulScalar(scale)
		v.grad2D.MulScalar(scale)
		v.grad3D.MulScalar(scale)
	}
}

func (v *Variable) SGD(lr float64) {
	grad1d := tensor.D1MulScalar(v.grad1D, lr)
	grad2d := tensor.D2MulScalar(v.grad2D, lr)
	grad3d := tensor.D3MulScalar(v.grad3D, lr)
	v.Param.D1.Sub(grad1d)
	v.Param.D2.Sub(grad2d)
	v.Param.D3.Sub(grad3d)
}

func (v *Variable) SetParam(param Param) {
	v.Param.D1.Copy(param.D1)
	v.Param.D2.Copy(param.D2)
	v.Param.D3.Copy(param.D3)
}

type Sequential struct {
	variable Variable
	Forwards layer1d.Forwards

	YLossCalculator      func(tensor.D1, tensor.D1) (float64, error)
	YLossDifferentiator func(tensor.D1, tensor.D1) (tensor.D1, error)

	Param1DLossCalculator func(tensor.D1) float64
	Param2DLossCalculator func(tensor.D2) float64
	Param3DLossCalculator func(tensor.D3) float64

	Param1DLossDifferentiator func(tensor.D1) tensor.D1
	Param2DLossDifferentiator func(tensor.D2) tensor.D2
	Param3DLossDifferentiator func(tensor.D3) tensor.D3

	MaxGradL2Norm float64
}

func NewSequential(variable Variable) Sequential {
	return Sequential{
		variable:                  variable,
		Param1DLossCalculator:     func(_ tensor.D1) float64 { return 0.0 },
		Param1DLossDifferentiator: tensor.NewD1ZerosLike,
		Param2DLossCalculator:     func(_ tensor.D2) float64 { return 0.0 },
		Param2DLossDifferentiator: tensor.NewD2ZerosLike,
		Param3DLossCalculator:     func(_ tensor.D3) float64 { return 0.0 },
		Param3DLossDifferentiator: tensor.NewD3ZerosLike,
	}
}

func NewLinearSum(xn int, output layer1d.Forward, c float64) (Sequential, Variable) {
	variable := Variable{
		Param: Param{
			D1: tensor.D1{0.0},
			D2: tensor.D2{tensor.NewD1Zeros(xn)},
		},
	}
	variable.Init()

	forwards := layer1d.Forwards{
		layer1d.NewLinearSumForward(variable.Param.D2[0], &variable.Param.D1[0], variable.GetGrad2D()[0], &variable.GetGrad1D()[0]),
		output,
	}

	linearSum := NewSequential(variable)
	linearSum.Forwards = forwards
	linearSum.YLossCalculator = ml1d.SumSquaredError
	linearSum.YLossDifferentiator = ml1d.SumSquaredErrorDerivative
	linearSum.Param2DLossCalculator = ml2d.L2Regularization(c)
	linearSum.Param2DLossDifferentiator = ml2d.L2RegularizationDerivative(c)
	return linearSum, variable
}

func NewIdentityLinearSum(xn int, c float64) (Sequential, Variable) {
	return NewLinearSum(xn, layer1d.IdentityForward, c)
}

func NewSigmoidLinearSum(xn int, c float64) (Sequential, Variable) {
	return NewLinearSum(xn, layer1d.SigmoidForward, c)
}

func NewAffine(ns []int, output layer1d.Forward, loss func(tensor.D1, tensor.D1) (float64, error), derivative func(tensor.D1, tensor.D1) (tensor.D1, error), c float64, r *rand.Rand) (Sequential, Variable) {
	layerN := len(ns) - 1
	param := Param{
		D1: make(tensor.D1, layerN-1),
		D2: make(tensor.D2, layerN),
		D3: make(tensor.D3, layerN),
	}

	for i := 0; i < layerN-1; i++ {
		param.D1[i] = 0.1
	}

	for i := 0; i < layerN; i++ {
		param.D2[i] = tensor.NewD1Zeros(ns[i+1])
	}

	for i := 0; i < layerN; i++ {
		param.D3[i] = tensor.NewD2He(ns[i], ns[i+1], r)
	}

	variable := Variable{
		Param: param,
	}
	variable.Init()

	var forwards layer1d.Forwards
	for i := 0; i < layerN; i++ {
		forwards = append(forwards, layer1d.NewAffineForward(
			variable.Param.D3[i],
			variable.Param.D2[i],
			variable.GetGrad3D()[i],
			variable.GetGrad2D()[i],
		))

		//最後は、出力層を追加する
		if i == (layerN - 1) {
			forwards = append(forwards, layer1d.SoftmaxForward)
		} else {
			forwards = append(forwards, layer1d.NewParamReLUForward(
				&variable.Param.D1[i],
				&variable.GetGrad1D()[i],
			))
		}
	}

	affine := NewSequential(variable)
	affine.Forwards = forwards
	affine.YLossCalculator = loss
	affine.YLossDifferentiator = derivative
	affine.Param3DLossCalculator = ml3d.L2Regularization(c)
	affine.Param3DLossDifferentiator = ml3d.L2RegularizationDerivative(c)
	return affine, variable
}

func NewSigmoidAffine(ns []int, c float64, r *rand.Rand) (Sequential, Variable) {
	return NewAffine(ns, layer1d.SigmoidForward, ml1d.SumSquaredError, ml1d.SumSquaredErrorDerivative, c, r)
}

func NewTanhAffine(ns []int, c float64, r *rand.Rand) (Sequential, Variable) {
	return NewAffine(ns, layer1d.TanhForward, ml1d.SumSquaredError, ml1d.SumSquaredErrorDerivative, c, r)
}

func NewSoftmaxAffine(ns []int, c float64, r *rand.Rand) (Sequential, Variable) {
	return NewAffine(ns, layer1d.SoftmaxForwardForCrossEntropy, ml1d.CrossEntropyError, ml1d.CrossEntropyErrorDerivative, c, r)
}

func (m *Sequential) Predict(x tensor.D1) (tensor.D1, error) {
	y, _, err := m.Forwards.Propagate(x)
	return y, err
}

func (m *Sequential) MeanLoss(x, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("バッチ数が一致しません。")
	}

	sum := m.Param1DLossCalculator(m.variable.Param.D1)
	sum += m.Param2DLossCalculator(m.variable.Param.D2)
	sum += m.Param3DLossCalculator(m.variable.Param.D3)
	sum *= float64(n)

	for i := range x {
		y, err := m.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		yLoss, err := m.YLossCalculator(y, t[i])
		if err != nil {
			return 0.0, err
		}
		sum += yLoss
	}
	mean := sum / float64(n)
	return mean, nil
}

func (m *Sequential) Accuracy(x, t tensor.D2) (float64, error) {
	n := len(x)
	if n != len(t) {
		return 0.0, fmt.Errorf("バッチ数が一致しません。")
	}

	correct := 0
	for i := range x {
		y, err := m.Predict(x[i])
		if err != nil {
			return 0.0, err
		}
		if omwslices.MaxIndex(y) == omwslices.MaxIndex(t[i]) {
			correct += 1
		}
	}
	return float64(correct) / float64(n), nil
}

func (m *Sequential) UpdateGrad(x, t tensor.D1) error {
	y, backwards, err := m.Forwards.Propagate(x)
	if err != nil {
		return err
	}

	dLdy, err := m.YLossDifferentiator(y, t)
	if err != nil {
		return err
	}

	_, err = backwards.Propagate(dLdy)
	if err != nil {
		return err
	}

	grad1D := m.Param1DLossDifferentiator(m.variable.Param.D1)
	grad2D := m.Param2DLossDifferentiator(m.variable.Param.D2)
	grad3D := m.Param3DLossDifferentiator(m.variable.Param.D3)

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

	if m.MaxGradL2Norm > 0.0 {
		m.variable.ClipGrads(m.MaxGradL2Norm)
	}
	return err
}

func (m *Sequential) SGD(lr float64) {
	m.variable.SGD(lr)
	m.variable.ResetGrad()
}

func (m *Sequential) ValidateBackwardAndNumericalGradientDifference(x, t tensor.D1) error {
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

	lossD3 := func(_ tensor.D3) float64 {
		loss, err := m.MeanLoss(tensor.D2{x}, tensor.D2{t})
		if err != nil {
			panic(err)
		}
		return loss
	}

	numGradD1 := ml1d.NumericalDifferentiation(m.variable.Param.D1, lossD1)
	numGradD2 := ml2d.NumericalDifferentiation(m.variable.Param.D2, lossD2)
	numGradD3 := ml3d.NumericalDifferentiation(m.variable.Param.D3, lossD3)
	if m.MaxGradL2Norm > 0.0 {
		m.variable.ClipGrads(m.MaxGradL2Norm)
	}
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

	var maxDiffD3 float64
	if len(diffD3) != 0 {
		maxDiffD3 = diffD3.MapFunc(math.Abs).MaxRow().MaxRow().Max()
	}

	fmt.Println("maxDiffD1 =", maxDiffD1)
	fmt.Println("maxDiffD2 =", maxDiffD2)
	fmt.Println("maxDiffD3 =", maxDiffD3)
	return nil
}
