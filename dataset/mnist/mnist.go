package mnist

import (
	"os"
	"gonum.org/v1/gonum/blas/blas32"
	ogob "github.com/sw965/omw/encoding/gob"
	"github.com/sw965/crow/blas32/tensor/3d"
)

var PATH = os.Getenv("GOPATH") + "/mnist/gob/"
var TRAIN_PATH = PATH + "train/"
var TEST_PATH = PATH + "test/"

//訓練データ
func LoadTrainFlatImages() ([]blas32.Vector, error) {
	return ogob.Load[[]blas32.Vector](TRAIN_PATH + "flat_img.gob")
}

func LoadTrainImages() ([]tensor3d.General, error) {
	return ogob.Load[[]tensor3d.General](TRAIN_PATH + "img.gob")
}

func LoadTrainLabels() ([]blas32.Vector, error) {
	return ogob.Load[[]blas32.Vector](TRAIN_PATH + "label.gob")
}


//テストデータ
func LoadTestFlatImages() ([]blas32.Vector, error) {
	return ogob.Load[[]blas32.Vector](TEST_PATH + "flat_img.gob")
}

func LoadTestImages() ([]tensor3d.General, error) {
	return ogob.Load[[]tensor3d.General](TEST_PATH + "img.gob")
}

func LoadTestLabels() ([]blas32.Vector, error) {
	return ogob.Load[[]blas32.Vector](TEST_PATH + "label.gob")
}