package model_test

import (
	"testing"
	"fmt"
	"github.com/sw965/crow/model/general"
	"github.com/sw965/crow/model/spsa"
	"github.com/sw965/crow/model/convert"
	"math/rand"
	orand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/dataset/mnist"
	tmath "github.com/sw965/crow/tensor/math"
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
	genModel.AppendInstanceNormalization(midN1)

	genModel.AppendDot(midN1, midN2, rng)
	genModel.AppendLeakyReLU(0.1)
	genModel.AppendInstanceNormalization(midN2)

	genModel.AppendDot(midN2, yn, rng)
	genModel.AppendLeakyReLU(0.1)
	genModel.AppendInstanceNormalization(yn)

	genModel.AppendSoftmaxForCrossEntropyLoss()
	genModel.SetCrossEntropyLossForSoftmax()

	/*
		InstanceNormlizationのGammaとBetaはそれぞれ1と0で初期化されてる。
		数値微分と解析的微分の誤差を測定する際には都合が悪い為、ランダムに書き換える。
	*/
	for _, gamma := range genModel.Parameter.Gammas {
		for j := range gamma.Data {
			gamma.Data[j] = rng.Float32()
		}
	}

	for _, bias := range genModel.Parameter.Biases {
		for j := range bias.Data {
			bias.Data[j] = rng.Float32()
		}
	}

	//SPSAで勾配を求めるモデルの生成
	spsaModel := spsa.Model{}
	spsaModel.AppendDot(xn, midN1, rng)
	spsaModel.AppendLeakyReLU(0.1)
	spsaModel.AppendInstanceNormalization(midN1)

	spsaModel.AppendDot(midN1, midN2, rng)
	spsaModel.AppendLeakyReLU(0.1)
	spsaModel.AppendInstanceNormalization(midN2)

	spsaModel.AppendDot(midN2, yn, rng)
	spsaModel.AppendLeakyReLU(0.1)
	spsaModel.AppendInstanceNormalization(yn)

	spsaModel.AppendSoftmax()

	//genModelとspsaModelを同じパラメーターにする
	spsaModel.Parameters = convert.GeneralParameterToSPSAParameters(&genModel.Parameter, spsaModel.LayerTypes)

	testSize := 100

	//mnistの生成
	trainXs, err := mnist.LoadTrainFlatImages()
	if err != nil {
		panic(err)
	}

	trainXs = trainXs[:testSize]

	trainYs, err := mnist.LoadTrainLabels()
	if err != nil {
		panic(err)
	}

	trainYs = trainYs[:testSize]

	lossFunc := func(model spsa.Model, workerIdx int) (float32, error) {
		sum := float32(0.0)
		for i, x := range trainXs {
			y, err := model.Predict(x)
			if err != nil {
				return 0.0, err
			}
			t := trainYs[i]
			loss := tmath.CrossEntropy(y, t)
			sum += loss
		}
		return sum / float32(testSize), nil
	}

	for i := 0; i < testSize; i++ {
		x := trainXs[i]
		genY, err := genModel.Predict(x)
		if err != nil {
			panic(err)
		}

		spsaY, err := spsaModel.Predict(x)
		if err != nil {
			panic(err)
		}

		fmt.Println("genY = ", genY)
		fmt.Println("spsaY =", spsaY)
		fmt.Println("")
	}

	//解析的微分と数値微分を計算して、誤差を計算する
	trueGrad, err := genModel.ComputeGrad(trainXs, trainYs, 12)
	if err != nil {
		panic(err)
	}

	numGrads, err := spsaModel.NumericalGrads(lossFunc)
	if err != nil {
		panic(err)
	}

	genNumGrad := convert.SPSAGradsToGeneralGrad(numGrads)
	fMaxDiffs, wMaxDiffs, gMaxDiffs, bMaxDiffs := trueGrad.CompareMaxDiff(genNumGrad)
	fmt.Println(fMaxDiffs)
	fmt.Println(wMaxDiffs)
	fmt.Println(gMaxDiffs)
	fmt.Println(bMaxDiffs)

	// 解析的微分とSPSAで複数回推定した勾配の平均との誤差を計算する
	p := 6
	rngs := make([]*rand.Rand, p)
	for i := range rngs {
		rngs[i] = orand.NewMt19937()
	}
	trialNum := 1280000
	total := genModel.Parameter.NewGradZerosLike()

	//一旦はSPSAで推定した勾配とのテストは実行しない
	return

	for i := 0; i < trialNum; i++ {
		spsaGrads, err := spsaModel.EstimateGrads(lossFunc, 0.01, rngs)
		if err != nil {
			panic(err)
		}
		genGrad := convert.SPSAGradsToGeneralGrad(spsaGrads)
		total.AxpyInPlace(1.0, genGrad)

		if i%12800 == 0 {
			avg := total.Clone()
			avg.ScalInPlace(1.0 / float32(i+1))
			fMaxDiffs, wMaxDiffs, gMaxDiffs, bMaxDiffs := trueGrad.CompareMaxDiff(avg)
			fmt.Println("i =", i)
			fmt.Println(fMaxDiffs)
			fmt.Println(wMaxDiffs)
			fmt.Println(gMaxDiffs)
			fmt.Println(bMaxDiffs)
			fmt.Println()
		}
	}
}