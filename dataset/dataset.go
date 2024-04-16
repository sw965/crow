package dataset

import (
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

var MNIST_PATH = omw.SW965_PATH + "mnist_json/"

type Mnist struct {
	TrainImg tensor.D2
	TrainLabel tensor.D2
	TestImg tensor.D2
	TestLabel tensor.D2
}

func LoadMnist() (Mnist, error) {
	trainImg, err := omw.LoadJSON[tensor.D2](MNIST_PATH + "train_img" + omw.JSON_EXTENSION)
	if err != nil {
		return Mnist{}, err
	}

	trainLabel, err := omw.LoadJSON[tensor.D2](MNIST_PATH + "train_onehot_label" + omw.JSON_EXTENSION)
	if err != nil {
		return Mnist{}, err
	}

	testImg, err := omw.LoadJSON[tensor.D2](MNIST_PATH + "test_img" + omw.JSON_EXTENSION)
	if err != nil {
		return Mnist{}, err
	}

	testLabel, err := omw.LoadJSON[tensor.D2](MNIST_PATH + "test_onehot_label" + omw.JSON_EXTENSION)
	return Mnist{TrainImg:trainImg, TrainLabel:trainLabel, TestImg:testImg, TestLabel:testLabel}, err
}