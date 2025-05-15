package spsa_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/dataset/mnist"
	"math/rand"
	orand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/model/spsa"
	tmath "github.com/sw965/crow/tensor/math"
)

func TestModel(t *testing.T) {
	xn := 784
	mid1N := 128
	mid2N := 32
	yn := 10

	rng := orand.NewMt19937()
	model := spsa.Model{}
	model.AppendDot(xn, mid1N, rng)
	model.AppendLeakyReLU(0.1)
	model.AppendInstanceNormalization(mid1N)

	model.AppendDot(mid1N, mid2N, rng)
	model.AppendLeakyReLU(0.1)
	model.AppendInstanceNormalization(32)

	model.AppendDot(mid2N, yn, rng)
	model.AppendLeakyReLU(0.1)
	model.AppendInstanceNormalization(10)

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

	lossFunc := func(model spsa.Model, workerIdx int) (float32, error) {
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
			loss := tmath.CrossEntropy(y, t)
			sum += loss
		}
		return sum / float32(miniBatchSize), nil
	}

	trainNum := 128000
	lr := float32(0.01)
	for i := 0; i < trainNum; i++ {
		grads, err := model.EstimateGrads(lossFunc, 0.2, rngs)
		if err != nil {
			panic(err)
		}
		model.Parameters.AxpyInPlaceGrads(-lr, grads)

		if i%512 == 0 {
			acc, err := model.Accuracy(testXs, testYs)
			if err != nil {
				panic(err)
			}
			fmt.Println("acc =", acc, "i =", i)
		}
	}
}
