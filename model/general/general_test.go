package general_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/model/general"
	"github.com/sw965/crow/dataset/mnist"
	orand "github.com/sw965/omw/math/rand"
	"gonum.org/v1/gonum/blas/blas32"
)

func TestModel(t *testing.T) {
	rng := orand.NewMt19937()
	model := general.Model{}
	model.AppendDot(784, 1280, rng)
	model.AppendLeakyReLU(0.1)
	model.AppendInstanceNormalization(1280)

	model.AppendDot(1280, 100, rng)
	model.AppendLeakyReLU(0.1)
	model.AppendInstanceNormalization(100)

	model.AppendDot(100, 10, rng)
	model.AppendLeakyReLU(0.1)
	model.AppendInstanceNormalization(10)

	model.AppendSoftmaxForCrossEntropyLoss()
	model.SetCrossEntropyLossForSoftmax()

	trainXs, err := mnist.LoadTrainFlatImages()
	if err != nil {
		panic(err)
	}

	trainYs, err := mnist.LoadTrainLabels()
	if err != nil {
		panic(err)
	}

	testXs, err := mnist.LoadTestFlatImages()
	if err != nil {
		panic(err)
	}

	testYs, err := mnist.LoadTestLabels()
	if err != nil {
		panic(err)
	}

	p := 12
	miniBatchSize := 512/p

	trainNum := 12800
	lr := float32(0.01)
	opt := general.NewMomentum(&model.Parameter)
	opt.LearningRate = lr

	for i := 0; i < trainNum; i++ {
		miniXs := make([]blas32.Vector, miniBatchSize)
		miniTs := make([]blas32.Vector, miniBatchSize)
		for j := 0; j < miniBatchSize; j++ {
			idx := rng.Intn(60000)
			miniXs[j] = trainXs[idx]
			miniTs[j] = trainYs[idx]
		}

		grad, err := model.ComputeGrad(miniXs, miniTs, p)
		if err != nil {
			panic(err)
		}
		opt.Optimize(&model, &grad, 0)
		//model.Parameter.AxpyGrad(-lr, &grad)

		if i%512 == 0 {
			acc, err := model.Accuracy(testXs[:1000], testYs[:1000])
			if err != nil {
				panic(err)
			}
			fmt.Println(i, acc)
		}
	}
}