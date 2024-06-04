package model1d

import (
	"fmt"
	"math"
	"math/rand"
	"github.com/sw965/crow/layer/1d"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
	"github.com/sw965/crow/mlfuncs/1d"
	"github.com/sw965/crow/mlfuncs/2d"
	"github.com/sw965/crow/mlfuncs/3d"
	omwslices "github.com/sw965/omw/slices"
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

type Sequential struct {
	variable Variable
	Forwards layer1d.Forwards

	YLossFunc func(tensor.D1, tensor.D1) (float64, error)
	YLossDerivative func(tensor.D1, tensor.D1) (tensor.D1, error)

	Param1DLossFunc func(tensor.D1)float64
	Param2DLossFunc func(tensor.D2)float64
	Param3DLossFunc func(tensor.D3)float64

	Param1DLossDerivative func(tensor.D1)tensor.D1
	Param2DLossDerivative func(tensor.D2)tensor.D2
	Param3DLossDerivative func(tensor.D3)tensor.D3

	L2NormGradClipThreshold float64
}

func NewSequential(variable Variable) Sequential {
	return Sequential{
		variable:variable,
		Param1DLossFunc:func(_ tensor.D1) float64 { return 0.0 },
		Param1DLossDerivative:tensor.NewD1ZerosLike,
		Param2DLossFunc:func(_ tensor.D2) float64 { return 0.0 },
		Param2DLossDerivative:tensor.NewD2ZerosLike,
		Param3DLossFunc:func(_ tensor.D3) float64 { return 0.0 },
		Param3DLossDerivative:tensor.NewD3ZerosLike,
	}
}

func NewStandardAffine(xn, h1, h2, yn int, c, threshold float64, r *rand.Rand) (Sequential, Variable) {
	variable := Variable{
		Param1D:tensor.D1{
			0.1,
			0.1,
		},

		Param2D:tensor.D2{
			tensor.NewD1Zeros(h1),
			tensor.NewD1Zeros(h2),
			tensor.NewD1Zeros(yn),
		},

		Param3D:tensor.D3{
			tensor.NewD2He(xn, h1, r),
			tensor.NewD2He(h1, h2, r),
			tensor.NewD2He(h2, yn, r),
		},
	}
	variable.Init()

	forwards := layer1d.Forwards{
		layer1d.NewAffineForward(variable.Param3D[0], variable.Param2D[0], variable.GetGrad3D()[0], variable.GetGrad2D()[0]),
		layer1d.NewParamReLUForward(&variable.Param1D[0], &variable.GetGrad1D()[0]),

		layer1d.NewAffineForward(variable.Param3D[1], variable.Param2D[1], variable.GetGrad3D()[1], variable.GetGrad2D()[1]),
		layer1d.NewParamReLUForward(&variable.Param1D[1], &variable.GetGrad1D()[1]),

		layer1d.NewAffineForward(variable.Param3D[2], variable.Param2D[2], variable.GetGrad3D()[2], variable.GetGrad2D()[2]),
		layer1d.NewSigmoidForward(),
	}

	affine := NewSequential(variable)
	affine.Forwards = forwards
	affine.YLossFunc = mlfuncs1d.SumSquaredError
	affine.YLossDerivative = mlfuncs1d.SumSquaredErrorDerivative
	affine.Param3DLossFunc = mlfuncs3d.L2Regularization(c)
	affine.Param3DLossDerivative = mlfuncs3d.L2RegularizationDerivative(c)
	affine.L2NormGradClipThreshold = threshold
	return affine, variable
}

func NewStandardLinearSum(xn int, c, threshold float64) (Sequential, Variable) {
	variable := Variable{
		Param1D:tensor.D1{0.0},
		Param2D:tensor.D2{tensor.NewD1Zeros(xn)},
	}
	variable.Init()

	forwards := layer1d.Forwards{
		layer1d.NewLinearSumForward(variable.Param2D[0], &variable.Param1D[0], variable.GetGrad2D()[0], &variable.GetGrad1D()[0]),
		layer1d.NewSigmoidForward(),
	}

	linearSum := NewSequential(variable)
	linearSum.Forwards = forwards
	linearSum.YLossFunc = mlfuncs1d.SumSquaredError
	linearSum.YLossDerivative = mlfuncs1d.SumSquaredErrorDerivative
	linearSum.Param2DLossFunc = mlfuncs2d.L2Regularization(c)
	linearSum.Param2DLossDerivative = mlfuncs2d.L2RegularizationDerivative(c)
	linearSum.L2NormGradClipThreshold = threshold
	return linearSum, variable
}

func (m *Sequential) Predict(x tensor.D1) (tensor.D1, error) {
	y, _, err := m.Forwards.Propagate(x)
	return y, err
}

func (m *Sequential) MeanLoss(x, t tensor.D2) (float64, error) {
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

func (m *Sequential) Accuracy(x, t tensor.D2) (float64, error) {
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

func (m *Sequential) SGD(x, t tensor.D1, lr float64) {
	m.UpdateGrad(x, t)
	m.variable.GradMulScaler(lr)
	m.variable.SGD()
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

	numGradD1 := mlfuncs1d.NumericalDifferentiation(m.variable.Param1D, lossD1)
	numGradD2 := mlfuncs2d.NumericalDifferentiation(m.variable.Param2D, lossD2)
	numGradD3 := mlfuncs3d.NumericalDifferentiation(m.variable.Param3D, lossD3)
	if m.L2NormGradClipThreshold > 0.0 {
		mlfuncs.ClipL2Norm(numGradD1, numGradD2, numGradD3, m.L2NormGradClipThreshold)
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