package model_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/model/general"
	"github.com/sw965/crow/model/spsa"
	orand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/dataset/mnist"
	"github.com/sw965/crow/blas32/vector"
	"math/rand"
	"github.com/chewxy/math32"
	"github.com/sw965/crow/blas32/tensor/2d"
)

func TestFullyConnectedModelGrad(t *testing.T) {
	rng := orand.NewMt19937()
	xn := 784
	midN1 := 10
	midN2 := 10
	yn := 10

	//解析的な微分をするモデルの生成
	genModel := general.Model{}
	genModel.AppendDot(xn, midN1, rng)
	genModel.AppendLeakyReLU(0.1)

	genModel.AppendDot(midN1, midN2, rng)
	genModel.AppendLeakyReLU(0.1)

	genModel.AppendDot(midN2, yn, rng)
	genModel.AppendLeakyReLU(0.1)

	genModel.AppendSoftmaxForCrossEntropyLoss()
	genModel.SetCrossEntropyLossForSoftmax()

	//SPSAで勾配を求めるモデルの生成
	spsaModel := spsa.Model{}
	spsaModel.AppendDot(xn, midN1, rng)
	spsaModel.AppendLeakyReLU(0.1)

	spsaModel.AppendDot(midN1, midN2, rng)
	spsaModel.AppendLeakyReLU(0.1)

	spsaModel.AppendDot(midN2, yn, rng)
	spsaModel.AppendLeakyReLU(0.1)

	spsaModel.AppendSoftmax()

	//genModelとspsaModelを同じパラメーターにする
	spsaModel.Parameters[0].Weight = tensor2d.Clone(genModel.Parameter.Weights[0])
	spsaModel.Parameters[2].Weight = tensor2d.Clone(genModel.Parameter.Weights[1])
	spsaModel.Parameters[4].Weight = tensor2d.Clone(genModel.Parameter.Weights[2])

	//mnistの生成
	trainXs, err := mnist.LoadTrainFlatImages()
	if err != nil {
		panic(err)
	}

	trainXs = trainXs[:1000]

	trainYs, err := mnist.LoadTrainLabels()
	if err != nil {
		panic(err)
	}

	trainYs = trainYs[:1000]
	dataN := len(trainXs)

	spsaModel.LossFunc = func(model *spsa.Model, workerIdx int) (float32, error) {
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
		return sum / float32(dataN), nil
	}

	trueGrad, err := genModel.ComputeGrad(trainXs, trainYs, 12)
	if err != nil {
		panic(err)
	}

	rngs := make([]*rand.Rand, 6)
	for i := range rngs {
		rngs[i] = orand.NewMt19937()
	}

	avgSpsaGrads := spsaModel.Parameters.NewGradsZerosLike()
	trialNum := 128000
	for i := 0; i < trialNum; i++ {
		spsaGrads, err := spsaModel.EstimateGrads(0.01, rngs)
		if err != nil {
			panic(err)
		}
		avgSpsaGrads.Axpy(1.0 / float32(trialNum), spsaGrads)

		if i%128 == 0 {
			trueW1 := trueGrad.Weights[0]
			spsaW1 := avgSpsaGrads[0].Weight
			maxDiff := float32(0.0)
			for j, tg := range trueW1.Data {
				diff := math32.Abs(tg - spsaW1.Data[j])
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			fmt.Println("maxDiff =", maxDiff, "i =", i)
		}
	}
}