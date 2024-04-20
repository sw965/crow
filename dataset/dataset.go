package dataset

import (
	"github.com/sw965/omw"
	"github.com/sw965/crow/tensor"
)

var FLAT_MNIST_PATH = omw.SW965_PATH + "crow/flat_mnist_json/"

type FlatMnist struct {
	TrainImg tensor.D2
	TrainLabel tensor.D2
	TestImg tensor.D2
	TestLabel tensor.D2
}

func LoadFlatMnist() (FlatMnist, error) {
	trainImg, err := omw.LoadJSON[tensor.D2](FLAT_MNIST_PATH + "train_img" + omw.JSON_EXTENSION)
	if err != nil {
		return Mnist{}, err
	}

	trainLabel, err := omw.LoadJSON[tensor.D2](FLAT_MNIST_PATH + "train_label" + omw.JSON_EXTENSION)
	if err != nil {
		return Mnist{}, err
	}

	testImg, err := omw.LoadJSON[tensor.D2](FLAT_MNIST_PATH + "test_img" + omw.JSON_EXTENSION)
	if err != nil {
		return Mnist{}, err
	}

	testLabel, err := omw.LoadJSON[tensor.D2](FLAT_MNIST_PATH + "test_label" + omw.JSON_EXTENSION)
	return Mnist{TrainImg:trainImg, TrainLabel:trainLabel, TestImg:testImg, TestLabel:testLabel}, err
}