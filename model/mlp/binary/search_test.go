package binary_test

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/sw965/crow/model/mlp/binary"
	"github.com/sw965/omw/mathx/bitsx"
)

func TestGridSearch(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))

	model := binary.Model{XRows: 1, XCols: 64}
	if err := model.AppendDenseLayer(32, rng); err != nil {
		t.Fatal(err)
	}
	if err := model.SetClassPrototypes(3, rng); err != nil {
		t.Fatal(err)
	}

	// 学習可能な合成データ:
	// クラス毎にランダムな中心パターンを作り、サンプルは中心の各ビットを5%で反転したもの。
	// 入力とラベルに明確な関係があるので、学習が機能していればチャンスレベル(1/3)を大きく超える。
	numClasses := 3
	centers := make(bitsx.Matrices, numClasses)
	for i := range numClasses {
		c, err := bitsx.NewRandMatrix(1, 64, 0, rng)
		if err != nil {
			t.Fatal(err)
		}
		centers[i] = c
	}

	n := 60
	xs := make(bitsx.Matrices, n)
	labels := make([]int, n)
	for i := range n {
		label := i % numClasses
		x := centers[label].Clone()
		for col := 0; col < x.Cols; col++ {
			if rng.Float64() < 0.05 {
				if err := x.Toggle(0, col); err != nil {
					t.Fatal(err)
				}
			}
		}
		xs[i] = x
		labels[i] = label
	}

	trainXs, trainLabels, valXs, valLabels, err := binary.SplitTrainValidation(xs, labels, 0.25, rng)
	if err != nil {
		t.Fatal(err)
	}
	if len(valXs) != 15 || len(trainXs) != 45 {
		t.Fatalf("unexpected split: train %d, val %d", len(trainXs), len(valXs))
	}

	// クローンの独立性確認用に、baseの重みをスナップショット
	dense := model.Backbone[0].(*binary.Dense)
	wSnapshot := slices.Clone(dense.W.Data)
	hSnapshot := slices.Clone(dense.H)

	space := binary.SearchSpace{
		LRs:        []float32{0.1, 0.5},
		GroupSizes: []int{4, 16},
	}

	epochs := 8
	results, err := binary.GridSearch(model, trainXs, trainLabels, valXs, valLabels, epochs, 8, 2, space, t.Logf)
	if err != nil {
		t.Fatal(err)
	}

	// 学習可能なデータなので、ベスト構成はチャンスレベル(1/3)を大きく超えるはず
	if results[0].BestValAcc < 0.6 {
		t.Fatalf("best val acc %.4f is too low: learning through GridSearch seems broken", results[0].BestValAcc)
	}

	if len(results) != 4 {
		t.Fatalf("expected 4 trials, got %d", len(results))
	}

	for _, r := range results {
		if len(r.ValAccs) != epochs {
			t.Fatalf("trial %s: expected %d val accs, got %d", r.Params, epochs, len(r.ValAccs))
		}
		if r.BestEpoch < 0 || r.BestEpoch >= epochs {
			t.Fatalf("trial %s: invalid BestEpoch %d", r.Params, r.BestEpoch)
		}
		if r.BestValAcc != slices.Max(r.ValAccs) {
			t.Fatalf("trial %s: BestValAcc %f != max(ValAccs) %f", r.Params, r.BestValAcc, slices.Max(r.ValAccs))
		}
	}

	// BestValAcc の降順でソートされている事
	for i := 1; i < len(results); i++ {
		if results[i-1].BestValAcc < results[i].BestValAcc {
			t.Fatalf("results are not sorted: %f < %f", results[i-1].BestValAcc, results[i].BestValAcc)
		}
	}

	// 全試行を回しても base モデルは変更されない事
	if !slices.Equal(wSnapshot, dense.W.Data) {
		t.Fatal("base model W was mutated by GridSearch")
	}
	if !slices.Equal(hSnapshot, dense.H) {
		t.Fatal("base model H was mutated by GridSearch")
	}
}
