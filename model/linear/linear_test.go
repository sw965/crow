package linear_test

import (
	"fmt"
	"testing"

	omwrand "github.com/sw965/omw/math/rand"
	"github.com/sw965/crow/model/linear"
	"github.com/sw965/crow/tensor"
)

func TestLearning(t *testing.T) {
	r := omwrand.NewMt19937()

	// モデル初期化: 入力2次元、2クラス分類
	// ※パラメータは、最初の要素が入力次元と一致している必要があるので、[2, 2]とする

	model := linear.Model{}
	model.Parameter = linear.NewInitParameter([]int{1, 1})
	model.SetSoftmaxForCrossEntropy()
	model.SetCrossEntropyError()

	// 学習用のデータセットを生成
	// 入力 x = [[a, b]] に対し、a > b なら ターゲット：[1, 0]、それ以外は [0, 1]
	sampleN := 10000
	xs := make(tensor.D3, sampleN)
	ts := make(tensor.D2, sampleN)

	for i := 0; i < sampleN; i++ {
		a := r.Float64()
		b := r.Float64()
		xs[i] = tensor.D2{tensor.D1{a}, tensor.D1{b}}
		if a > b {
			ts[i] = tensor.D1{1.0, 0.0}
		} else {
			ts[i] = tensor.D1{0.0, 1.0}
		}
	}

	momentum := linear.NewMomentum(&model)
	momentum.LearningRate = 0.00001
	teacher := &linear.MiniBatchTeacher{
		Inputs:xs,
		Labels:ts,
		MiniBatchSize:32,
		Epoch:        1000,
		Optimizer:momentum.Optimizer,
		Parallel:     4,
	}

	// SGDにより学習を実施
	if err := teacher.Teach(&model, r); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// 学習済みモデルで訓練データに対する正解率を評価
	acc, err := model.Accuracy(xs, ts)
	if err != nil {
		t.Fatalf("Accuracy evaluation failed: %v", err)
	}
	t.Logf("Training accuracy: %.2f%%", acc*100)

	if acc < 0.9 {
		t.Fatalf("Training accuracy too low: %.2f%%", acc*100)
	}
	fmt.Println(model.Parameter)
}

func TestSPSA(t *testing.T) {
	// 乱数生成器（固定シードで再現性を持たせる）
	r := omwrand.NewMt19937()

	// モデルを初期化
	// 今回は weight の形状は 2×1、bias は長さ2 とする
	model := linear.Model{}
	model.Parameter.Weight = tensor.D2{
		tensor.D1{1.0},
		tensor.D1{2.0},
	}
	model.Parameter.Bias = tensor.D1{3.0, 4.0}

	// 損失関数として、二乗誤差関数（ターゲットは以下の値とする）
	targetWeight := tensor.D2{
		tensor.D1{0.5},
		tensor.D1{1.5},
	}
	targetBias := tensor.D1{2.5, 3.5}
	model.LossFunc = func(m *linear.Model) (float64, error) {
		loss := 0.0
		for i, row := range m.Parameter.Weight {
			for j, w := range row {
				diff := w - targetWeight[i][j]
				loss += 0.5 * diff * diff
			}
		}
		for i, b := range m.Parameter.Bias {
			diff := b - targetBias[i]
			loss += 0.5 * diff * diff
		}
		return loss, nil
	}

	// 解析的な勾配は、単純に現在のパラメータとターゲットの差
	trueGradWeight := tensor.D2{
		tensor.D1{1.0 - 0.5}, // 0.5
		tensor.D1{2.0 - 1.5}, // 0.5
	}
	trueGradBias := tensor.D1{
		3.0 - 2.5, // 0.5
		4.0 - 3.5, // 0.5
	}

	// SPSAによる勾配推定を多数回行い平均値を求める
	iterations := 10000
	// sumGradW, sumGradB は同じ形状で初期化
	sumGradW := tensor.NewD2Zeros(len(model.Parameter.Weight), len(model.Parameter.Weight[0]))
	sumGradB := tensor.NewD1Zeros(len(model.Parameter.Bias))
	for i := 0; i < iterations; i++ {
		grad, err := model.EstimateGradBySPSA(1e-3, r)
		if err != nil {
			t.Fatalf("EstimateGradBySPSA failed: %v", err)
		}
		// grad.Weight, grad.Bias に対して要素ごとに加算
		for rowIdx, row := range grad.Weight {
			// SPSA実装内部で grad.Weight が初期化されていない場合は
			// ここでゼロ行列と同じ形状に初期化する必要があります。
			if len(sumGradW) <= rowIdx || len(sumGradW[rowIdx]) == 0 {
				t.Fatalf("grad.Weight の形状が不正です")
			}
			for colIdx, v := range row {
				sumGradW[rowIdx][colIdx] += v
			}
		}
		for i, v := range grad.Bias {
			sumGradB[i] += v
		}
	}
	// 平均値を計算
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

	// 推定された平均勾配が解析的勾配と十分近いかをチェック（許容誤差 tol）
	tol := 1e-2
	for i, row := range avgGradW {
		for j, v := range row {
			if diff := v - trueGradWeight[i][j]; diff < -tol || diff > tol {
				t.Fatalf("Weight 勾配不一致 (%d,%d): 推定値 %v, 解析値 %v", i, j, v, trueGradWeight[i][j])
			}
		}
	}
	for i, v := range avgGradB {
		if diff := v - trueGradBias[i]; diff < -tol || diff > tol {
			t.Fatalf("Bias 勾配不一致 (%d): 推定値 %v, 解析値 %v", i, v, trueGradBias[i])
		}
	}
}