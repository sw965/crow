package main

import (
	"os"
	ojson "github.com/sw965/omw/encoding/json"
	ogob "github.com/sw965/omw/encoding/gob"
	"github.com/sw965/crow/tensor"

)

var (
	JSON_PATH       = os.Getenv("GOPATH") + "/mnist/json/"
	JSON_TRAIN_PATH = JSON_PATH + "train/"
	JSON_TEST_PATH  = JSON_PATH + "test/"

	GOB_PATH       = os.Getenv("GOPATH") + "/mnist/gob/"
	GOB_TRAIN_PATH = GOB_PATH + "train/"
	GOB_TEST_PATH  = GOB_PATH + "test/"
)

const (
	CHANNELS = 1
	ROWS = 28
	COLS = 28
)

func saveGob(jsonPath, gobPath string) {
	jsonXs, err := ojson.Load[[][]float32](jsonPath + "flat_img.json")
	if err != nil {
		panic(err)
	}

	gobVectorXs := make(tensor.D1Slice, len(jsonXs))
	for i := range gobVectorXs {
		x := jsonXs[i]
		gobVectorXs[i] = tensor.D1{
			N:len(x),
			Inc:1,
			Data:x,
		}
	}

	err = ogob.Save(&gobVectorXs, gobPath + "flat_img.gob")
	if err != nil {
		panic(err)
	}

	gobD3Xs := make(tensor.D3Slice, len(jsonXs))
	for i := range gobD3Xs {
		gobD3Xs[i] = tensor.NewD3Zeros(CHANNELS, ROWS, COLS)
		gobD3Xs[i].Data = jsonXs[i]
	}

	err = ogob.Save(&gobD3Xs, gobPath + "img.gob")
	if err != nil {
		panic(err)
	}

	jsonLabels, err := ojson.Load[[][]float32](jsonPath + "label.json")
	if err != nil {
		panic(err)
	}

	gobLabels := make(tensor.D1Slice, len(jsonLabels))
	for i := range gobLabels {
		label := jsonLabels[i]
		gobLabels[i] = tensor.D1{
			N:len(label),
			Inc:1,
			Data:label,
		}
	}

	err = ogob.Save(&gobLabels, gobPath + "label.gob")
	if err != nil {
		panic(err)
	}
}

func main() {
	saveGob(JSON_TRAIN_PATH, GOB_TRAIN_PATH)
	saveGob(JSON_TEST_PATH, GOB_TEST_PATH)
}