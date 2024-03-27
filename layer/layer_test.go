package layer_test

import (
	"testing"
	"fmt"
	"math/rand"
	"github.com/sw965/crow/layer"
	"github.com/sw965/crow/tensor"
	omwrand "github.com/sw965/omw/rand"
)

type TestModel struct {
	Affine1 layer.D1Affine
	Prelu1 layer.D1ParametricReLU

	Affine2 layer.D1Affine
	Prelu2 layer.D1ParametricReLU

	Tanh layer.D1Tanh
	MSE layer.D1MeanSquaredError
}

func NewTestModel(xSize, ySize int, r *rand.Rand) TestModel {
	hidden1Size := 5

	affine1 := layer.NewD1Affine(xSize, hidden1Size, -0.01, 0.01, r)
	prelu1 := layer.NewD1ParametricReLU(hidden1Size, -0.01, 0.01, r)
	affine2 := layer.NewD1Affine(hidden1Size, ySize, -0.01, 0.01, r)
	prelu2 := layer.NewD1ParametricReLU(ySize, -0.01, 0.01, r)
	tanh := layer.D1Tanh{}
	mse := layer.D1MeanSquaredError{}

	return TestModel{
		Affine1:affine1,
		Prelu1:prelu1,
		Affine2:affine2,
		Prelu2:prelu2,
		Tanh:tanh,
		MSE:mse,
	}
}

func (model *TestModel) Predict(x tensor.D1) tensor.D1 {
	u1, err := model.Affine1.Forward(x)
	if err != nil {
		panic(err)
	}

	u2, err := model.Prelu1.Forward(u1)
	if err != nil {
		panic(err)
	}

	u3, err := model.Affine2.Forward(u2)
	if err != nil {
		panic(err)
	}

	u4, err := model.Prelu2.Forward(u3)
	if err != nil {
		panic(err)
	}

	y := model.Tanh.Forward(u4)
	return y
}

func (model *TestModel) Loss(x, t tensor.D1) float64 {
	y := model.Predict(x)
	return model.MSE.Forward(y, t)
}

func (model *TestModel) Backward() {
	chain1 := model.MSE.Backward()
	fmt.Println("c1", chain1)

	chain2, err := model.Tanh.Backward(chain1)
	if err != nil {
		panic(err)
	}
	fmt.Println("c2", chain2)

	chain3 := model.Prelu2.Backward(chain2)
	fmt.Println("c3", chain3)

	chain4, err := model.Affine2.Backward(chain3)
	if err != nil {
		panic(err)
	}
	fmt.Println("c4", chain4)

	chain5 := model.Prelu1.Backward(chain4)
	_, err = model.Affine1.Backward(chain5)
	if err != nil {
		panic(err)
	}
	fmt.Println("c5", chain5)

	fmt.Println(model.Affine1.Dw)
	fmt.Println(model.Affine2.Dw)
	fmt.Println(model.Prelu1.Dalpha)
}

func (model *TestModel) NumericalGradient(x, t tensor.D1) {
	h := 0.001
	grads := tensor.NewD2Zeros(len(model.Affine1.W), len(model.Affine1.W[0]))
	for i := range model.Affine1.W {
		for j := range model.Affine1.W[i] {
			tmp := model.Affine1.W[i][j]

			model.Affine1.W[i][j] = tmp + h
			y1 := model.Loss(x, t)
	
			model.Affine1.W[i][j] = tmp - h
			y2 := model.Loss(x, t)
	
			grads[i][j] = (y1 - y2) / (h * 2)
			model.Affine1.W[i][j] = tmp
		}
	}
	fmt.Println(grads)

	grads = tensor.NewD2Zeros(len(model.Affine2.W), len(model.Affine2.W[0]))
	for i := range model.Affine2.W {
		for j := range model.Affine2.W[i] {
			tmp := model.Affine2.W[i][j]

			model.Affine2.W[i][j] = tmp + h
			y1 := model.Loss(x, t)
	
			model.Affine2.W[i][j] = tmp - h
			y2 := model.Loss(x, t)
	
			grads[i][j] = (y1 - y2) / (h * 2)
			model.Affine2.W[i][j] = tmp
		}
	}
	fmt.Println(grads)

	grads1d := make(tensor.D1, model.Prelu1.Len)
	for i := range model.Prelu1.Alpha {
		tmp := model.Prelu1.Alpha[i]

		model.Prelu1.Alpha[i] = tmp + h
		y1 := model.Loss(x, t)
	
		model.Prelu1.Alpha[i] = tmp - h
		y2 := model.Loss(x, t)
	
		grads1d[i] = (y1 - y2) / (h * 2)
		model.Prelu1.Alpha[i] = tmp
	}
	fmt.Println(grads1d)
}

func TestLayer(t *testing.T) {
	r := omwrand.NewMt19937()
	x := tensor.NewD1Random(5, 1.0, 5.0, r)
	model := NewTestModel(len(x), 1, r)

	target := tensor.D1{0.0}
	y := model.Predict(x)
	fmt.Println("y =", y)
	loss := model.Loss(x, target)
	fmt.Println("loss = ", loss)
	model.Backward()
	fmt.Println()
	model.NumericalGradient(x, target)
}