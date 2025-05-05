package mlp_test

import (
	"fmt"
	cmath "github.com/sw965/crow/math"
	"github.com/sw965/crow/model/mlp"
	"testing"
	//oslices "github.com/sw965/omw/slices"
	"github.com/sw965/crow/dataset/mnist"
	orand "github.com/sw965/omw/math/rand"
	"math/rand"
)

func TestModel(t *testing.T) {
	rng := orand.NewMt19937()
	model := mlp.Model{}
	model.AppendAffine(784, 128, rng)
	model.AppendLeakyReLU(0.1)

	model.AppendAffine(128, 32, rng)
	model.AppendLeakyReLU(0.1)

	model.AppendAffine(32, 10, rng)
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

	miniBatchSize := 512

	rngs := make([]*rand.Rand, 4)
	for i := range rngs {
		rngs[i] = orand.NewMt19937()
	}

	model.LossFunc = func(model *mlp.Model, workerIdx int) (float32, error) {
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
			loss, err := cmath.CrossEntropyError(y, t)
			if err != nil {
				return 0.0, err
			}
			sum += loss
		}
		ceeLoss := sum / float32(miniBatchSize)
		l2Loss := 0.001 * model.Parameters.L2Norm()
		return ceeLoss + l2Loss, nil
	}

	trainNum := 128000
	lr := float32(0.01)
	c := float32(0.2)

	for i := 0; i < trainNum; i++ {
		grads, err := model.EstimateGradsBySPSA(c, rngs)
		if err != nil {
			panic(err)
		}

		model.Parameters.AxpyGrads(-lr, grads)

		if i%512 == 0 {
			acc, err := model.Accuracy(testXs, testYs)
			if err != nil {
				panic(err)
			}
			fmt.Println(i, acc)
		}
	}
}
