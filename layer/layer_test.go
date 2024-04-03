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
	PRReLU1 layer.D1PRReLU
	
	Affine2 layer.D1Affine
	PRReLU2 layer.D1PRReLU

	Affine3 layer.D1Affine
	PRReLU3 layer.D1PRReLU
}

func (model *Model) Predict(x tensor.D1) (tensor.D1, Backwards, error) {
	backwards := make(layer.Backwards, 10)
	u1, backwards, err := model.Affine1.Forward(x, backwards)
	if err != nil {
		return 0.0, Backwards{}, err
	}

	u2, backwards, err := model.PRReLU1.Forward(u1, backwards)
	if err != nil {
		return 0.0, err
	}

	u3, backwards, err := model.Affine2.Forward(u2, backwards)
	if err != nil {
		return 0.0, err
	}

	u4, backwards, err := model.PRReLU2.Forward(u3, backwards)
	if err != nil {
		return 0.0, err
	}

	y, backwards, err := layer.D1TanhForward(u4, backwards)
	return y, backwards, err
}

func (model *TestModel) Loss(x, t tensor.D1) (tensor.D1, Backwards, error) {
	y, backwards, err := model.Predict(x)
	 layer.D1MeanSquaredError(y, t)
}

func TestMnist(t *testing.T) {

}