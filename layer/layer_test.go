package layer_test

import (
	"testing"
	"fmt"
	"math/rand"
	"github.com/sw965/crow/mlfuncs"
	"github.com/sw965/crow/layer"
	"github.com/sw965/crow/tensor"
	"github.com/sw965/omw"
)

const (
	MNIST_IMG_SIZE = 784
)

type TestModel struct {
	Affine1 layer.D1Affine
	Activation1 layer.D1PReLU
	
	Affine2 layer.D1Affine
	Activation2 layer.D1PReLU

	Affine3 layer.D1Affine
	Activation3 layer.D1PReLU

	Output layer.Forward
}

func NewTestModel(h1, h2 int, r *rand.Rand) TestModel {
	output := 10
	return TestModel{
		Affine1:layer.NewD1Affine(MNIST_IMG_SIZE, h1, r),
		Activation1 :layer.NewD1PReLU(h1, 0.0, 0.01, r),
		//Activation1:layer.D1ReLUForward,

		Affine2:layer.NewD1Affine(h1, h2, r),
		Activation2:layer.NewD1PReLU(h2, 0.0, 0.01, r),
		//Activation2:layer.D1ReLUForward,

		Affine3:layer.NewD1Affine(h2, output, r),
		Activation3:layer.NewD1PReLU(output, 0.0, 0.01, r),
		//Activation3:layer.D1ReLUForward,

		//Sigmoid:layer.D1SigmoidForward,
		Output:layer.NormalizeToProbDistForward,
	}
}

func (model *TestModel) Predict(x tensor.D1) (tensor.D1, layer.Backwards, error) {
	fs := layer.Forwards{
		model.Affine1.Forward,
		model.Activation1.Forward,

		model.Affine2.Forward,
		model.Activation2.Forward,

		model.Affine3.Forward,
		model.Activation3.Forward,

		model.Output,
	}
	return fs.Run(x)
}

func (model *TestModel) YAndLoss(x, t tensor.D1) (tensor.D1, float64, layer.Backwards, error) {
	y, backwards, err := model.Predict(x)
	loss, backwards, err := layer.D1MeanSquaredErrorForward(y, t, backwards)
	loss += mlfuncs.D2L2Regularization(model.Affine1.W, model.Affine1.Lambda)
	loss += mlfuncs.D2L2Regularization(model.Affine2.W, model.Affine2.Lambda)
	loss += mlfuncs.D2L2Regularization(model.Affine3.W, model.Affine3.Lambda)
	return y, loss, backwards, err
}

func (model *TestModel) UpdateGrad(x, t tensor.D1) error {
	_, _, backwards, err := model.YAndLoss(x, t)
	if err != nil {
		return err
	}

	_, err = backwards.Run(tensor.NewD1Ones(len(t)))
	return err
}

func (model *TestModel) Train(lr float64) {
	model.Affine1.Train(lr)
	model.Affine2.Train(lr)
	model.Affine3.Train(lr)

	model.Activation1.Train(lr)
	model.Activation2.Train(lr)
	model.Activation3.Train(lr)
}

func (model *TestModel) SWA(old *TestModel, wScale float64) (TestModel, error) {
	affine1, err := model.Affine1.SWA(&old.Affine1, wScale)
	if err != nil {
		return TestModel{}, err
	}

	affine2, err := model.Affine2.SWA(&old.Affine2, wScale)
	if err != nil {
		return TestModel{}, err
	}

	affine3, err := model.Affine3.SWA(&old.Affine3, wScale)
	if err != nil {
		return TestModel{}, err
	}

	act1, err := model.Activation1.SWA(&old.Activation1, wScale)
	if err != nil {
		return TestModel{}, err
	}

	act2, err := model.Activation2.SWA(&old.Activation2, wScale)
	if err != nil {
		return TestModel{}, err
	}

	act3, err := model.Activation3.SWA(&old.Activation3, wScale)
	return TestModel{
		Affine1:affine1,
		Activation1:act1,
		//Activation1:model.Activation1,

		Affine2:affine2,
		Activation2:act2,
		//Activation2:model.Activation2,
	
		Affine3:affine3,
		Activation3:act3,
		//Activation3:model.Activation3,

		Output:model.Output,
	}, err
}


func TestMnist(t *testing.T) {
	r := omw.NewMt19937()
	model := NewTestModel(1024, 512, r)
	mnistPath := omw.SW965_PATH + "mnist_json/"
	trainImg, err := omw.LoadJSON[tensor.D2](mnistPath + "train_img" + omw.JSON_EXTENSION)
	if err != nil {
		panic(err)
	}

	trainLabel, err := omw.LoadJSON[tensor.D2](mnistPath + "train_onehot_label" + omw.JSON_EXTENSION)
	if err != nil {
		panic(err)
	}

	testImg, err := omw.LoadJSON[tensor.D2](mnistPath + "test_img" + omw.JSON_EXTENSION)
	if err != nil {
		panic(err)
	}

	testLabel, err := omw.LoadJSON[tensor.D2](mnistPath + "test_onehot_label" + omw.JSON_EXTENSION)
	if err != nil {
		panic(err)
	}

	lr := 0.01
	trainNum := 500000
	trainSize := 60000
	testSize := 10000
	scalar := 0.1
	oldModel, err := model.SWA(&model, 0.5)
	if err != nil {
		panic(err)
	}

	for i := 0; i < trainNum; i++ {
		idx := r.Intn(trainSize)
		err := model.UpdateGrad(trainImg[idx], trainLabel[idx])
		if err != nil {
			panic(err)
		}
		model.Train(lr)

		if i%16 == 0 {
			model, err = model.SWA(&oldModel, scalar)
			if err != nil {
				panic(err)
			}

			oldModel, err = model.SWA(&model, 0.5)
			if err != nil {
				panic(err)
			}
		}

		if i%512 == 0 {
			fmt.Println("i = ", i)
			testNum := 512
			sumLoss := 0.0
			predicts := make([]int, testNum)
			labels := make([]int, testNum)
			for j := 0; j < testNum; j++ {
				idx := r.Intn(testSize)
				label := testLabel[idx]
				y, loss, _, err := model.YAndLoss(testImg[idx], label)
				if err != nil {
					panic(err)
				}
				sumLoss += loss
				idxs := omw.MaxIndices(y)
				if len(idxs) == 0 {
					predicts[j] = 0
				} else {
					predicts[j] = idxs[0]
				}
				labels[j] = omw.MaxIndices(label)[0]
			}

			ac := 0
			for k := range predicts {
				if predicts[k] == labels[k] {
					ac += 1
				}
			}

			fmt.Println(sumLoss / float64(testNum))
			fmt.Println(float64(ac) / float64(testNum))
			fmt.Println("")
		}
	}
}