package model_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/dataset"
	"github.com/sw965/crow/model"
	"github.com/sw965/crow/tensor"
	omwrand "github.com/sw965/omw/math/rand"
	omwjson "github.com/sw965/omw/json"
)

func TestCNN(t *testing.T) {
	r := omwrand.NewMt19937()
	cnn := model.Sequential{}
	cnn.SetCrossEntropyError()

	cnn.AppendConvLayer(2, 2, 1, 10, r)
	cnn.AppendLeakyReLULayer(0.1)

	cnn.AppendConvLayer(2, 2, 10, 10, r)
	cnn.AppendLeakyReLULayer(0.1)

	cnn.AppendGAPLayer()
	cnn.AppendSoftmaxForCrossEntropyLayer()

	mnist, err := dataset.LoadFlatMnist()
	if err != nil {
		panic(err)
	}

	mnist.TrainImg = make(tensor.D2, 0, 0)
	mnist.TestImg = make(tensor.D2, 0, 0)

	trainImg, err := omwjson.Load[tensor.D4]("C:/Go/project/foo/main/trainx.json")
	if err != nil {
		panic(err)
	}

	testImg, err := omwjson.Load[tensor.D4]("C:/Go/project/foo/main/testx.json")
	if err != nil {
		panic(err)
	}

	fmt.Println(len(trainImg), len(trainImg[0]), len(trainImg[0][0]), len(trainImg[0][0][0]))
	fmt.Println(len(testImg), len(testImg[0]), len(testImg[0][0]), len(testImg[0][0][0]))

	mbc := model.MiniBatchConfig{
		BatchSize:16,
		Epoch:1,
		LearningRate:0.01,
		Parallel:4,
	}

	//fmt.Println("len= ", len(trainImg), len(testImg))

	for i := 0; i < 128; i++ {
		err := cnn.Train(trainImg, mnist.TrainLabel, &mbc, r)
		if err != nil {
			panic(err)
		}

		fmt.Println("1エポック終了")

		loss, err := cnn.MeanLoss(testImg[:2000], mnist.TestLabel[:2000])
		if err != nil {
			panic(err)
		}

		accuracy, err := cnn.Accuracy(testImg[:2000], mnist.TestLabel[:2000])
		if err != nil {
			panic(err)
		}

		fmt.Println("i =", i, "loss =", loss, "acc =", accuracy)
	}
}