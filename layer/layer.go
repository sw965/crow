package layer

import (
	"math/rand"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
)

type D1Affine struct {
	X tensor.D1
	W tensor.D2
	B tensor.D1
	Dw tensor.D2
	Db tensor.D1
}

func NewD1Affine(r, c int, min, max float64, random *rand.Rand) D1Affine {
	return D1Affine{
		W:tensor.NewD2Random(r, c, min, max, random),
		B:tensor.NewD1Random(c, min, max, random),
	}
}

func (affine *D1Affine) Forward(x tensor.D1) (tensor.D1, error) {
	affine.X = x
	dot, err := tensor.D2{x}.DotProduct(affine.W)
	if err != nil {
		return tensor.D1{}, err
	}
	y, err := dot[0].Add(affine.B)
	return y, err
}

func (affine *D1Affine) Backward(chain tensor.D1) (tensor.D1, error) {
	//∂L/∂x
	dx, err := tensor.D2{chain}.DotProduct(affine.W.Transpose())
	if err != nil {
		return tensor.D1{}, err
	}

	// ∂L/∂w
	affine.Dw, err = tensor.D2{affine.X}.Transpose().DotProduct(tensor.D2{chain})

	if err != nil {
		return tensor.D1{}, err
	}

	// ∂L/∂b
	affine.Db = chain
	return dx[0], nil
}

type D1ParametricReLU struct {
	X tensor.D1
	Alpha tensor.D1
	Dalpha tensor.D1
	Len int
}

func NewD1ParametricReLU(n int, max, min float64, random *rand.Rand) D1ParametricReLU {
	return D1ParametricReLU{
		Alpha:tensor.NewD1Random(n, max, min, random),
		Dalpha:make(tensor.D1, n),
		Len:n,
	}
}

func (prelu *D1ParametricReLU) Forward(x tensor.D1) (tensor.D1, error) {
	prelu.X = x
	return mlfuncs.D1ParametricReLU(x, prelu.Alpha)
}

func (prelu *D1ParametricReLU) Backward(chain tensor.D1) tensor.D1 {
	dx := make(tensor.D1, prelu.Len)
	for i := range dx {
		if prelu.X[i] >= 0 {
			// 正のXに対しては、∂L/∂x はそのまま
			dx[i] = chain[i]
			// 正のXに対しては、Alphaの勾配は0
			prelu.Dalpha[i] = 0
		} else {
			dx[i] = prelu.Alpha[i] * chain[i]
			prelu.Dalpha[i] = prelu.X[i] * chain[i]
		}
	}
	return dx
}

type D1Tanh struct {
	Y tensor.D1
}

func (tanh *D1Tanh) Forward(x tensor.D1) tensor.D1 {
	y := mlfuncs.D1Tanh(x)
	tanh.Y = y
	return y
}

func (tanh *D1Tanh) Backward(chain tensor.D1) (tensor.D1, error) {	
	dydx := mlfuncs.D1TanhGrad(tanh.Y)
	// ∂L/∂x
	dx, err := dydx.Mul(chain)
	return dx, err
}

type D1MeanSquaredError struct {
	Y tensor.D1
	T tensor.D1
}

func (mse *D1MeanSquaredError) Forward(y, t tensor.D1) float64 {
	mse.Y = y
	mse.T = t
	return mlfuncs.D1MeanSquaredError(y, t)
}

func (mse *D1MeanSquaredError) Backward() tensor.D1 {
	return mlfuncs.D1MeanSquaredErrorDerivative(mse.Y, mse.T)
}
