package dataset

import (
	opath "github.com/sw965/omw/path"
	ojson "github.com/sw965/omw/json"
	"github.com/sw965/crow/tensor"
)

var FLAT_MNIST_PATH = opath.SW965 + "crow/flat_mnist_json/"

type FlatMnist struct {
	TrainImg tensor.D2
	TrainLabel tensor.D2
	TestImg tensor.D2
	TestLabel tensor.D2
}

func LoadFlatMnist() (FlatMnist, error) {
	trainImg, err := ojson.Load[tensor.D2](FLAT_MNIST_PATH + "train_img" + ojson.EXTENSION)
	if err != nil {
		return FlatMnist{}, err
	}

	trainLabel, err := ojson.Load[tensor.D2](FLAT_MNIST_PATH + "train_label" + ojson.EXTENSION)
	if err != nil {
		return FlatMnist{}, err
	}

	testImg, err := ojson.Load[tensor.D2](FLAT_MNIST_PATH + "test_img" + ojson.EXTENSION)
	if err != nil {
		return FlatMnist{}, err
	}

	testLabel, err := ojson.Load[tensor.D2](FLAT_MNIST_PATH + "test_label" + ojson.EXTENSION)
	return FlatMnist{TrainImg:trainImg, TrainLabel:trainLabel, TestImg:testImg, TestLabel:testLabel}, err
}