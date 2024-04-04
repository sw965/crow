package layer_test

import (
	"testing"
	"fmt"
	"math/rand"
	"github.com/sw965/crow/layer"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/crow/mlfuncs"
	"github.com/sw965/omw"
)

const (
	MNIST_IMG_SIZE = 784
)

type TestModel struct {
	Affine1 layer.D1Affine
	PRReLU1 layer.D1PRReLU
	
	Affine2 layer.D1Affine
	PRReLU2 layer.D1PRReLU

	Affine3 layer.D1Affine
	PRReLU3 layer.D1PRReLU

	Output layer.Forward
}

func NewTestModel(h1, h2 int, r *rand.Rand) TestModel {
	bMin := 3.0 / 8.0
	bMax := 1.0 / 3.0

	return TestModel{
		Affine1:layer.NewD1Affine(MNIST_IMG_SIZE, h1, -0.1, 0.1, r),
		PRReLU1:layer.NewD1PRReLU(h1, 0, 0.01, bMin, bMax, r),

		Affine2:layer.NewD1Affine(h1, h2, -0.1, 0.1, r),
		PRReLU2:layer.NewD1PRReLU(h2, 0, 0.01, bMin, bMax, r),

		Affine3:layer.NewD1Affine(h2, 1, -0.1, 0.1, r),
		PRReLU3:layer.NewD1PRReLU(1, 0, 0.01, bMin, bMax, r),

		Output:layer.D1TanhForward,
	}
}

func (model *TestModel) Predict(x tensor.D1, isTrain bool) (tensor.D1, layer.Backwards, error) {
	model.PRReLU1.IsTrain = isTrain
	model.PRReLU2.IsTrain = isTrain
	model.PRReLU3.IsTrain = isTrain

	fs := layer.Forwards{
		model.Affine1.Forward,
		model.PRReLU1.Forward,
		model.Affine2.Forward,
		model.PRReLU2.Forward,
		model.Affine3.Forward,
		model.PRReLU3.Forward,
		model.Output,
	}
	return fs.Run(x)
}

func (model *TestModel) YAndLoss(x, t tensor.D1, lambda float64, isTrain bool) (tensor.D1, float64, layer.Backwards, error) {
	y, backwards, err := model.Predict(x, isTrain)
	loss := mlfuncs.D1MeanSquaredError(y, t)
	loss += mlfuncs.D2L2Regularization(model.Affine1.W, lambda)
	loss += mlfuncs.D2L2Regularization(model.Affine2.W, lambda)
	loss += mlfuncs.D2L2Regularization(model.Affine3.W, lambda)
	return y, loss, backwards, err
}

func (model *TestModel) Accuracy(x tensor.D1, t float64) (float64, error) {
	y, _, err := model.Predict(x, false)
	diff := mlfuncs.ScalarTanhToSigmoidScale(y[0]) - t
	if diff > 0 && diff < 0.1 {
		return 1.0, err
	} else {
		return 0.0, err
	}
}

func (model *TestModel) UpdateGrad(x, t tensor.D1, lambda float64) error {
	y, _, backwards, err := model.YAndLoss(x, t, lambda, true)
	if err != nil {
		return err
	}

	chain := mlfuncs.D1MeanSquaredErrorDerivative(y, t)
	_, err = backwards.Run(chain)
	if err != nil {
		return err
	}

	dw1 := model.Affine1.Dw
	model.Affine1.Dw, err = dw1.Add(mlfuncs.D2L2RegularizationDerivative(dw1, lambda))
	if err != nil {
		return err
	}

	dw2 := model.Affine2.Dw
	model.Affine2.Dw, err = dw2.Add(mlfuncs.D2L2RegularizationDerivative(dw2, lambda))
	if err != nil {
		return err
	}

	dw3 := model.Affine3.Dw
	model.Affine3.Dw, err = dw3.Add(mlfuncs.D2L2RegularizationDerivative(dw3, lambda))
	return err
}

func (model *TestModel) Train(lr float64, batchSize int) error {
	err := model.Affine1.SGD(lr, batchSize)
	if err != nil {
		return err
	}

	err = model.PRReLU1.SGD(lr, batchSize)
	if err != nil {
		return err
	}

	err = model.Affine2.SGD(lr, batchSize)
	if err != nil {
		return err
	}

	err = model.PRReLU2.SGD(lr, batchSize)
	if err != nil {
		return err
	}

	err = model.Affine3.SGD(lr, batchSize)
	if err != nil {
		return err
	}

	err = model.PRReLU3.SGD(lr, batchSize)
	return err
}

func TestMnist(t *testing.T) {
	r := omw.NewMt19937()
	model := NewTestModel(64, 16, r)
	mnistPath := omw.SW965_PATH + "mnist_json/"
	trainImg, err := omw.LoadJSON[tensor.D2](mnistPath + "train_img" + omw.JSON_EXTENSION)
	if err != nil {
		panic(err)
	}

	trainLabel, err := omw.LoadJSON[tensor.D1](mnistPath + "train_label" + omw.JSON_EXTENSION)
	if err != nil {
		panic(err)
	}

	testImg, err := omw.LoadJSON[tensor.D2](mnistPath + "test_img" + omw.JSON_EXTENSION)
	if err != nil {
		panic(err)
	}

	testLabel, err := omw.LoadJSON[tensor.D1](mnistPath + "test_label" + omw.JSON_EXTENSION)
	if err != nil {
		panic(err)
	}

	lr := 0.1
	lambda := 0.001
	trainNum := 5000

	trainSize := 60000
	testSize := 10000
	batchSize := 128

	for i := 0; i < trainNum; i++ {
		for n := 0; n < batchSize; n++ {
			idx := r.Intn(trainSize)
			err := model.UpdateGrad(trainImg[idx], tensor.D1{mlfuncs.ScalarSigmoidToTanhScale(trainLabel[idx]/10.0)}, lambda)
			if err != nil {
				panic(err)
			}
		}
		model.Train(lr, batchSize)

		if i%100 == 0 {
			testNum := 128
			sum := 0.0
			as := 0.0
			for j := 0; j < testNum; j++ {
				idx := r.Intn(testSize)
				_, loss, _, err := model.YAndLoss(testImg[idx], tensor.D1{mlfuncs.ScalarSigmoidToTanhScale(testLabel[idx]/10.0)}, lambda, false)
				if err != nil {
					panic(err)
				}
				sum += loss

				a, err := model.Accuracy(testImg[idx], testLabel[idx]/10.0)
				if err != nil {
					panic(err)
				}
				as += a
			}
			fmt.Println(sum / float64(testNum), as / float64(testNum))
		}
	}
}