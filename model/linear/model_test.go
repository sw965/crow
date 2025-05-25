package linear_test

import (
	"fmt"
	"github.com/sw965/crow/model/linear"
	orand "github.com/sw965/omw/math/rand"
	"math/rand"
	"runtime"
	"testing"
	"slices"
)

const scale = 1.0

func makeRandWeight(n int, rng *rand.Rand) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(rng.Float64() * scale)
	}
	return w
}

func makeRandBias(n int, rng *rand.Rand) []float32 {
	b := make([]float32, n)
	for i := range b {
		b[i] = float32(rng.Float64() * scale)
	}
	return b
}

func makeRandInputs(batchSize, weightN, rows int,
	sparsePercent float64, rng *rand.Rand) linear.Inputs {

	inputs := make(linear.Inputs, 0, batchSize)

	for i := 0; i < batchSize; i++ {
		// []linear.Entries を rows 個確保
		input := make(linear.Input, rows)

		nonEmpty := false // 最低 1 行に Entry が入ったか判定
		for r := 0; r < rows; r++ {
			if rng.Float64() >= sparsePercent { // 生成するかどうか
				entry := linear.Entry{
					X:           float32(rng.Float64() * scale),
					WeightIndex: rng.Intn(weightN),
				}
				input[r] = append(input[r], entry)
				nonEmpty = true
			}
		}

		if nonEmpty {
			inputs = append(inputs, input)
		}
	}
	return inputs
}

func makeRandLabels(batchSize, n int, rng *rand.Rand) [][]float32 {
	ts := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		t := make([]float32, n)
		sum := float32(0.0)
		for j := 0; j < n; j++ {
			v := float32(rng.Float64())
			t[j] = v
			sum += v
		}
		// 確率分布に正規化
		for j := 0; j < n; j++ {
			t[j] /= sum
		}
		ts[i] = t
	}
	return ts
}

func TestU(t *testing.T) {
	param := linear.Parameter{
		Weight:[]float32{
			1.0, 2.0, 3.0, 5.0,
		},

		Bias:[]float32{
			10.0, 20.0,
		},
	}

	model := linear.Model{
		Parameter:param,
		OutputLayer:linear.NewIdentityLayer(),
		BiasIndices:[]int{0, 1, 0},
	}

	rows := 3
	input := make(linear.Input, rows)
	input[0] = linear.Entries{
		linear.Entry{
			X:1.0,
			WeightIndex:0,
		},

		linear.Entry{
			X:3.0,
			WeightIndex:1,
		},

		linear.Entry{
			X:2.0,
			WeightIndex:2,
		},
	}

	input[1] = linear.Entries{
		linear.Entry{
			X:3.0,
			WeightIndex:3,
		},

		linear.Entry{
			X:1.0,
			WeightIndex:1,
		},
	}

	input[2] = linear.Entries{
		linear.Entry{
			X:6.0,
			WeightIndex:2,
		},
	}

	u := model.U(input)
	if !slices.Equal([]float32{23.0, 37.0, 28.0}, u) {
		t.Errorf("テスト失敗")
	}
}

func TestSoftmaxCrossEntropyLossGrad(t *testing.T) {
	rng := orand.NewMt19937()
	weightN := 500000
	outputN := 5
	w := makeRandWeight(weightN, rng)
	b := makeRandBias(3, rng)

	model := linear.Model{
		Parameter: linear.Parameter{Weight: w, Bias: b},
		//OutputLayer:linear.NewSoftmaxLayer(),
		OutputLayer: linear.NewIdentityLayer(),
		BiasIndices:[]int{0, 1, 0, 1, 2},
	}
	//lossLayer := linear.NewCrossEntropyLossLayer()
	lossLayer := linear.NewMSELoss()

	batchSize := 16
	inputs := makeRandInputs(batchSize, weightN, outputN, 0.0, rng)
	labels := makeRandLabels(batchSize, outputN, rng)

	p := runtime.NumCPU()
	trueGrad, err := model.ComputeGrad(inputs, labels, lossLayer, p)
	if err != nil {
		panic(err)
	}

	lossFunc := func(model linear.Model, workerIdx int) (float32, error) {
		sum := float32(0.0)
		for i, input := range inputs {
			y := model.Predict(input)
			t := labels[i]
			loss := lossLayer.Func(y, t)
			sum += loss
		}
		return sum / float32(batchSize), nil
	}

	grad, err := model.PartialDifferentiation(lossFunc, p)
	if err != nil {
		panic(err)
	}

	grad.Axpy(-1.0, trueGrad)
	maxWDiff, maxBDiff := grad.MaxAbs()
	fmt.Println(maxWDiff, maxBDiff)

	rngs := make([]*rand.Rand, p)
	for i := 0; i < p; i++ {
		rngs[i] = orand.NewMt19937()
	}

	spsaTrialNum := 1280
	spsaGrads := make(linear.GradBuffers, spsaTrialNum)
	for i := 0; i < spsaTrialNum; i++ {
		spsaGrad, err := model.EstimateGradBySPSA(0.1, lossFunc, rngs)
		if err != nil {
			panic(err)
		}
		spsaGrads[i] = spsaGrad
	}

	spsaAvgGrad := spsaGrads.Average()
	spsaAvgGrad.Axpy(-1.0, trueGrad)
	maxWDiff, maxBDiff = spsaAvgGrad.MaxAbs()
	fmt.Println(maxWDiff, maxBDiff)
}
