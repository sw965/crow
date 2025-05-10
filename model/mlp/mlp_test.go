package mlp_test

import (
	"fmt"
	//cmath "github.com/sw965/crow/math"
	"github.com/sw965/crow/model/mlp"
	"testing"
	//oslices "github.com/sw965/omw/slices"
	"github.com/sw965/crow/dataset/mnist"
	orand "github.com/sw965/omw/math/rand"
	"gonum.org/v1/gonum/blas/blas32"
	//"math/rand"
)

func TestModel(t *testing.T) {
	rng := orand.NewMt19937()
	model := mlp.Model{}
	model.AppendAffine(784, 784, rng)
	model.AppendLeakyReLU(0.1)

	model.AppendAffine(784, 32, rng)
	model.AppendLeakyReLU(0.1)

	model.AppendAffine(32, 10, rng)
	model.AppendLeakyReLU(0.1)

	model.AppendOutputSoftmaxAndSetCrossEntropyLoss()

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

	p := 6
	miniBatchSize := 128*p

	trainNum := 12800
	//lr := float32(0.01)
	opt := mlp.NewMomentum(model.Parameters)

	for i := 0; i < trainNum; i++ {
		miniXs := make([]blas32.Vector, miniBatchSize)
		miniTs := make([]blas32.Vector, miniBatchSize)
		for j := 0; j < miniBatchSize; j++ {
			idx := rng.Intn(60000)
			miniXs[j] = trainXs[idx]
			miniTs[j] = trainYs[idx]
		}

		grads, err := model.ComputeGradByTeacher(miniXs, miniTs, rng, p)
		if err != nil {
			panic(err)
		}
		//model.Parameters.AxpyGrads(-lr, grads)
		opt.Optimizer(&model, grads)

		if i%512 == 0 {
			acc, err := model.Accuracy(testXs[:1000], testYs[:1000])
			if err != nil {
				panic(err)
			}
			fmt.Println(i, acc)
		}
	}
}