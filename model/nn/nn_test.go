package nn_test

// import (
// 	"testing"
// 	"fmt"
// 	"github.com/sw965/crow/dataset"
// 	"github.com/sw965/crow/model/nn"
// 	omwrand "github.com/sw965/omw/math/rand"
// )

// func TestNN(t *testing.T) {
// 	r := omwrand.NewMt19937()
// 	model := nn.FullyConnected{}
// 	model.SetCrossEntropyError()

// 	model.AppendFullyConnectedLayer(784, 128, r)
// 	model.AppendLeakyReLULayer(0.1)

// 	model.AppendFullyConnectedLayer(128, 10, r)
// 	model.AppendLeakyReLULayer(0.1)

// 	model.AppendSoftmaxForCrossEntropyLayer()

// 	mnist, err := dataset.LoadFlatMnist()
// 	if err != nil {
// 		panic(err)
// 	}

// 	mbc := nn.MiniBatchConfig{
// 		BatchSize:16,
// 		Epoch:1,
// 		LearningRate:0.01,
// 		Parallel:4,
// 	}

// 	for i := 0; i < 128; i++ {
// 		err := model.Train(mnist.TrainImg, mnist.TrainLabel, &mbc, r)
// 		if err != nil {
// 			panic(err)
// 		}

// 		fmt.Println("1エポック終了")

// 		loss, err := model.MeanLoss(mnist.TestImg[:2000], mnist.TestLabel[:2000])
// 		if err != nil {
// 			panic(err)
// 		}

// 		accuracy, err := model.Accuracy(mnist.TestImg[:2000], mnist.TestLabel[:2000])
// 		if err != nil {
// 			panic(err)
// 		}

// 		fmt.Println("i =", i, "loss =", loss, "acc =", accuracy)
// 	}
// }