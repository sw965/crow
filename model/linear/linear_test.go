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

	model := linear.Sum{}
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

	config := &linear.MiniBatchConfig{
		LearningRate: 0.001,
		BatchSize:    32,
		Epoch:        1000,
		Parallel:     4,
	}

	// SGDにより学習を実施
	if err := model.TrainBySGD(xs, ts, config, r); err != nil {
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