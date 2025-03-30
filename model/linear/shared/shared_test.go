package shared_test

import (
	"fmt"
	"testing"
	"github.com/sw965/crow/model/linear/shared"
	"github.com/sw965/crow/tensor"
	omwrand "github.com/sw965/omw/math/rand"
)

func TestLearning(t *testing.T) {
	// 乱数生成器
	r := omwrand.NewMt19937()

	// モデル初期化（2クラス分類用：入力は2要素からなる各サンプル）
	var model shared.Model

	// Parameter.Weight は 2×1 の行列、Bias は長さ2 のベクトルとして初期化
	weight := make([][]*float64, 2)
	for i := 0; i < 2; i++ {
		weight[i] = make([]*float64, 1)
		weight[i][0] = new(float64)
		*weight[i][0] = 1.0
	}
	bias := make([]*float64, 2)
	for i := 0; i < 2; i++ {
		bias[i] = new(float64)
		*bias[i] = 0.0
	}
	model.Parameter = shared.Parameter{
		Weight: weight,
		Bias:   bias,
	}

	// 出力層の活性化関数と損失関数を設定（softmaxと交差エントロピー誤差）
	model.SetSoftmaxForCrossEntropy()
	model.SetCrossEntropyError()

	// 学習用データセットの生成
	// 各サンプルは tensor.D2 として、1要素ずつの2行からなる行列となる
	// a, b の値に対し、 a > b なら [1, 0]、それ以外は [0, 1] をターゲットとする
	sampleN := 10000
	xs := make(tensor.D3, sampleN)
	ts := make(tensor.D2, sampleN)
	for i := 0; i < sampleN; i++ {
		a := r.Float64()
		b := r.Float64()
		xs[i] = tensor.D2{
			tensor.D1{a},
			tensor.D1{b},
		}
		if a > b {
			ts[i] = tensor.D1{1.0, 0.0}
		} else {
			ts[i] = tensor.D1{0.0, 1.0}
		}
	}

	// Momentum オプティマイザの初期化
	momentum := shared.NewMomentum(&model)
	momentum.LearningRate = 0.00001

	teacher := &shared.MiniBatchTeacher{
		Inputs:        xs,
		Labels:        ts,
		MiniBatchSize: 32,
		Epoch:         1000,
		Optimizer:     momentum.Optimizer,
		Parallel:      4,
	}

	// 教師あり学習を実施
	if err := teacher.Teach(&model, r); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// 訓練データに対する正解率評価
	acc, err := model.Accuracy(xs, ts)
	if err != nil {
		t.Fatalf("Accuracy evaluation failed: %v", err)
	}
	t.Logf("Training accuracy: %.2f%%", acc*100)

	if acc < 0.9 {
		t.Fatalf("Training accuracy too low: %.2f%%", acc*100)
	}
	fmt.Println("Trained Parameter:", model.Parameter)
}

func TestSPSA(t *testing.T) {
	r := omwrand.NewMt19937()

	// モデル初期化：重みは 2×1、バイアスは長さ2 とする
	var model shared.Model

	weight := make([][]*float64, 2)
	weight[0] = make([]*float64, 1)
	weight[0][0] = new(float64)
	*weight[0][0] = 1.0
	weight[1] = make([]*float64, 1)
	weight[1][0] = new(float64)
	*weight[1][0] = 2.0

	bias := make([]*float64, 2)
	bias[0] = new(float64)
	*bias[0] = 3.0
	bias[1] = new(float64)
	*bias[1] = 4.0

	model.Parameter = shared.Parameter{
		Weight: weight,
		Bias:   bias,
	}

	// ターゲットパラメーター（解析的に勾配が求まるように設定）
	targetWeight := tensor.D2{
		tensor.D1{0.5},
		tensor.D1{1.5},
	}
	targetBias := tensor.D1{2.5, 3.5}

	// LossFunc は二乗誤差関数として、各パラメーターとターゲットとの差の二乗和を返す
	model.LossFunc = func(m *shared.Model) (float64, error) {
		loss := 0.0
		for i, row := range m.Parameter.Weight {
			for j, ptr := range row {
				diff := *ptr - targetWeight[i][j]
				loss += 0.5 * diff * diff
			}
		}
		for i, ptr := range m.Parameter.Bias {
			diff := *ptr - targetBias[i]
			loss += 0.5 * diff * diff
		}
		return loss, nil
	}

	// 解析的な勾配（現在のパラメーターとターゲットとの差）
	trueGradWeight := tensor.D2{
		tensor.D1{1.0 - 0.5}, // 0.5
		tensor.D1{2.0 - 1.5}, // 0.5
	}
	trueGradBias := tensor.D1{
		3.0 - 2.5, // 0.5
		4.0 - 3.5, // 0.5
	}

	// SPSA による勾配推定を多数回実施し平均値を求める
	iterations := 10000
	sumGradW := tensor.NewD2Zeros(len(model.Parameter.Weight), len(model.Parameter.Weight[0]))
	sumGradB := tensor.NewD1Zeros(len(model.Parameter.Bias))
	for i := 0; i < iterations; i++ {
		grad, err := model.EstimateGradBySPSA(1e-3, r)
		if err != nil {
			t.Fatalf("EstimateGradBySPSA failed: %v", err)
		}
		// 各要素を加算
		for rowIdx, row := range grad.Weight {
			for colIdx, v := range row {
				sumGradW[rowIdx][colIdx] += v
			}
		}
		for i, v := range grad.Bias {
			sumGradB[i] += v
		}
	}
	// 平均値計算
	avgGradW := make(tensor.D2, len(sumGradW))
	for i, row := range sumGradW {
		avgGradW[i] = make(tensor.D1, len(row))
		for j, v := range row {
			avgGradW[i][j] = v / float64(iterations)
		}
	}
	avgGradB := make(tensor.D1, len(sumGradB))
	for i, v := range sumGradB {
		avgGradB[i] = v / float64(iterations)
	}

	t.Logf("解析的勾配 Weight: %v", trueGradWeight)
	t.Logf("SPSA勾配（平均） Weight: %v", avgGradW)
	t.Logf("解析的勾配 Bias: %v", trueGradBias)
	t.Logf("SPSA勾配（平均） Bias: %v", avgGradB)

	// 推定された勾配が解析的勾配と十分に近いか確認（許容誤差 tol）
	tol := 1e-2
	for i, row := range avgGradW {
		for j, v := range row {
			diff := v - trueGradWeight[i][j]
			if diff < -tol || diff > tol {
				t.Fatalf("Weight 勾配不一致 (%d,%d): 推定値 %v, 解析値 %v", i, j, v, trueGradWeight[i][j])
			}
		}
	}
	for i, v := range avgGradB {
		diff := v - trueGradBias[i]
		if diff < -tol || diff > tol {
			t.Fatalf("Bias 勾配不一致 (%d): 推定値 %v, 解析値 %v", i, v, trueGradBias[i])
		}
	}
}