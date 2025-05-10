package spsa_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/dataset/mnist"
	"math/rand"
	orand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/model/spsa"
	"github.com/sw965/crow/blas32/vector"
)

func TestModel(t *testing.T) {
	rng := orand.NewMt19937()
	model := spsa.Model{}
	model.AppendDot(784, 1280, rng)
	model.AppendLeakyReLU(0.1)

	model.AppendDot(1280, 100, rng)
	model.AppendLeakyReLU(0.1)

	model.AppendDot(100, 10, rng)
	model.AppendLeakyReLU(0.1)

	model.AppendSoftmax()

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
	rngs := make([]*rand.Rand, p)
	for i := range rngs {
		rngs[i] = orand.NewMt19937()
	}
	miniBatchSize := 128

	model.LossFunc = func(model *spsa.Model, workerIdx int) (float32, error) {
		rng := rngs[workerIdx]
		sum := float32(0.0)
		for i := 0; i < miniBatchSize; i++ {
			idx := rng.Intn(60000)
			x := trainXs[idx]
			y, err := model.Predict(x)
			if err != nil {
				return 0.0, err
			}
			t := trainYs[idx]
			loss, err := vector.CrossEntropy(y, t)
			if err != nil {
				return 0.0, err
			}
			sum += loss
		}
		return sum / float32(miniBatchSize), nil
	}

	trainNum := 1280
	lr := float32(0.01)
	for i := 0; i < trainNum; i++ {
		grads, err := model.EstimateGrads(0.01, rngs)
		if err != nil {
			panic(err)
		}
		model.Parameters.AxpyGrads(-lr, grads)

		if i%512 == 0 {
			acc, err := model.Accuracy(testXs, testYs)
			if err != nil {
				panic(err)
			}
			fmt.Println("acc =", acc, "i =", i)
		}
	}
}
