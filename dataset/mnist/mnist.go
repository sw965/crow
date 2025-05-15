package mnist

import (
	"os"
	ogob "github.com/sw965/omw/encoding/gob"
	"github.com/sw965/crow/tensor"
)

var PATH = os.Getenv("GOPATH") + "/mnist/gob/"
var TRAIN_PATH = PATH + "train/"
var TEST_PATH = PATH + "test/"

//訓練データ
func LoadTrainFlatImages() (tensor.D1Slice, error) {
	return ogob.Load[tensor.D1Slice](TRAIN_PATH + "flat_img.gob")
}

func LoadTrainImages() (tensor.D3Slice, error) {
	return ogob.Load[tensor.D3Slice](TRAIN_PATH + "img.gob")
}

func LoadTrainLabels() (tensor.D1Slice, error) {
	return ogob.Load[tensor.D1Slice](TRAIN_PATH + "label.gob")
}

//テストデータ
func LoadTestFlatImages() (tensor.D1Slice, error) {
	return ogob.Load[tensor.D1Slice](TEST_PATH + "flat_img.gob")
}

func LoadTestImages() (tensor.D3Slice, error) {
	return ogob.Load[tensor.D3Slice](TEST_PATH + "img.gob")
}

func LoadTestLabels() (tensor.D1Slice, error) {
	return ogob.Load[tensor.D1Slice](TEST_PATH + "label.gob")
}