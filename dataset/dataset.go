package dataset

import (
	omwpath "github.com/sw965/omw/path"
	omwjson "github.com/sw965/omw/json"
	"github.com/sw965/crow/tensor"
)

var FLAT_MNIST_PATH = omwpath.SW965 + "crow/flat_mnist_json/"

type FlatMnist struct {
	TrainImg tensor.D2
	TrainLabel tensor.D2
	TestImg tensor.D2
	TestLabel tensor.D2
}

func LoadFlatMnist() (FlatMnist, error) {
	trainImg, err := omwjson.Load[tensor.D2](FLAT_MNIST_PATH + "train_img" + omwjson.EXTENSION)
	if err != nil {
		return FlatMnist{}, err
	}

	trainLabel, err := omwjson.Load[tensor.D2](FLAT_MNIST_PATH + "train_label" + omwjson.EXTENSION)
	if err != nil {
		return FlatMnist{}, err
	}

	testImg, err := omwjson.Load[tensor.D2](FLAT_MNIST_PATH + "test_img" + omwjson.EXTENSION)
	if err != nil {
		return FlatMnist{}, err
	}

	testLabel, err := omwjson.Load[tensor.D2](FLAT_MNIST_PATH + "test_label" + omwjson.EXTENSION)
	return FlatMnist{TrainImg:trainImg, TrainLabel:trainLabel, TestImg:testImg, TestLabel:testLabel}, err
}