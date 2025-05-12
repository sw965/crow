package spsa_test

import (
	"testing"
	"fmt"
	omath "github.com/sw965/omw/math"
	orand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/model/spsa"
	"github.com/sw965/crow/dataset/mnist"
	"github.com/sw965/crow/blas32/vector"
	"math/rand"
)

func Test(t *testing.T) {
	rng := orand.NewMt19937()
	model := spsa.Model{}
	model.AppendDot(784, 128, rng)
	model.AppendLeakyReLU(0.1)
	model.AppendInstanceNormalization(128)

	model.AppendDot(128, 32, rng)
	model.AppendLeakyReLU(0.1)
	model.AppendInstanceNormalization(32)

	model.AppendDot(32, 10, rng)
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

	miniBatchSize := 10
	trainXs = trainXs[:miniBatchSize]
	trainYs = trainYs[:miniBatchSize]

	lossFunc := func(model *spsa.Model, workerIdx int) (float32, error) {
		sum := float32(0.0)
		for i, x := range trainXs {
			y, err := model.Predict(x)
			if err != nil {
				return 0.0, err
			}
			t := trainYs[i]

			loss, err := vector.CrossEntropy(y, t)
			if err != nil {
				return 0.0, err
			}
			sum += loss
		}
		return sum / float32(miniBatchSize), nil
	}

	numGrads, err := model.NumericalGrads(lossFunc)
	if err != nil {
		panic(err)
	}

	p := 6
	rngs := make([]*rand.Rand, p)
	for i := range rngs {
		rngs[i] = orand.NewMt19937()
	}

	trialNum := 128000
	avg := model.Parameters.NewGradsZerosLike()
	for i := 0; i < trialNum; i++ {
		spsaGrads, err := model.EstimateGrads(lossFunc, 0.01, rngs)
		if err != nil {
			panic(err)
		}
		avg.Axpy(1.0 / float32(trialNum), spsaGrads)

		if i%1280 == 0 {
			maxDiff := float32(0.0)
			for i, numGrad := range numGrads {
				spsaGrad := spsaGrads[i]
				spsaGrad.Axpy(1.0, &numGrad)
				abs := spsaGrad.Abs()
				maxes := make([]float32, 0, 3)
				if abs.Weight.Rows != 0 {
					maxes = append(maxes, omath.Max(abs.Weight.Data...))
				}

				if abs.Gamma.N != 0 {
					maxes = append(maxes, omath.Max(abs.Gamma.Data...))
				}

				if abs.Bias.N != 0 {
					maxes = append(maxes, omath.Max(abs.Bias.Data...))
				}

				if len(maxes) != 0 {
					diff := omath.Max(maxes...)
					if diff > maxDiff {
						maxDiff = diff
					}
				}
			}
			fmt.Println("i =", i, "maxDiff =", maxDiff)
		}
	}
}