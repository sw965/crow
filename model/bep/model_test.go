package bep_test

import (
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/sw965/crow/model/bep"
	"github.com/sw965/omw/mathx/bitsx"
)

func TestModelSetValues(t *testing.T) {
	t.Run("正常_昇順に等間隔", func(t *testing.T) {
		model, _ := newTestModel(t)
		if err := model.SetValues(0.0, 1.0); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}

		// クラス数4なので、0.0から1.0を3等分した値になる
		want := []float32{0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}
		if len(model.Values) != len(want) {
			t.Fatalf("len(Values)の不一致: got = %d, want = %d", len(model.Values), len(want))
		}
		for i := range want {
			if math.Abs(float64(model.Values[i]-want[i])) > 0.0001 {
				t.Errorf("Values[%d]の不一致: got = %f, want = %f", i, model.Values[i], want[i])
			}
		}

		if !slices.IsSorted(model.Values) {
			t.Errorf("Valuesが昇順ではない: %v", model.Values)
		}
	})

	t.Run("正常_Tanh用の範囲", func(t *testing.T) {
		model, _ := newTestModel(t)
		if err := model.SetTanhValues(); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if model.Values[0] != -1.0 || model.Values[len(model.Values)-1] != 1.0 {
			t.Errorf("値域の不一致: got = [%f, %f], want = [-1.0, 1.0]", model.Values[0], model.Values[len(model.Values)-1])
		}
	})

	t.Run("異常_minがmax以上", func(t *testing.T) {
		model, _ := newTestModel(t)
		err := model.SetValues(1.0, 1.0)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "min < max") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})

	t.Run("異常_Prototypes未設定", func(t *testing.T) {
		model := bep.Model{}
		err := model.SetValues(0.0, 1.0)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestModelValueToLabel(t *testing.T) {
	model := bep.Model{Values: []float32{0.0, 0.25, 0.5, 0.75, 1.0}}

	tests := []struct {
		name string
		val  float32
		want int
	}{
		{name: "正常_一致", val: 0.5, want: 2},
		{name: "正常_最近傍", val: 0.3, want: 1},
		{name: "正常_境界_下限未満", val: -100.0, want: 0},
		{name: "正常_境界_上限超過", val: 100.0, want: 4},
		{name: "正常_等距離は小さい方", val: 0.125, want: 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := model.ValueToLabel(tc.val)
			if got != tc.want {
				t.Errorf("値の不一致: got = %d, want = %d", got, tc.want)
			}
		})
	}

	t.Run("準正常_Valuesが空", func(t *testing.T) {
		empty := bep.Model{}
		if got := empty.ValueToLabel(0.5); got != 0 {
			t.Errorf("値の不一致: got = %d, want = 0", got)
		}
	})

	t.Run("性質_線形探索と一致", func(t *testing.T) {
		// 二分探索の結果が、素朴な線形探索(最小距離・同距離なら小さいインデックス)と一致する事
		rng := rand.New(rand.NewPCG(3, 4))
		values := make([]float32, 16)
		v := float32(0.0)
		for i := range values {
			v += rng.Float32()
			values[i] = v
		}
		m := bep.Model{Values: values}

		linearSearch := func(val float32) int {
			bestIdx := 0
			minDiff := float32(math.Abs(float64(val - values[0])))
			for i := 1; i < len(values); i++ {
				diff := float32(math.Abs(float64(val - values[i])))
				if diff < minDiff {
					minDiff = diff
					bestIdx = i
				}
			}
			return bestIdx
		}

		for range 1000 {
			val := rng.Float32() * v
			got := m.ValueToLabel(val)
			want := linearSearch(val)
			if got != want {
				t.Fatalf("線形探索と不一致: val = %f, got = %d, want = %d", val, got, want)
			}
		}
	})
}

func TestModelPredict(t *testing.T) {
	model, rng := newTestModel(t)
	if err := model.SetSigmoidValues(); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	x := newTestInput(t, rng)

	t.Run("性質_logitsの範囲", func(t *testing.T) {
		logits, err := model.PredictLogits(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if len(logits) != 4 {
			t.Fatalf("len(logits)の不一致: got = %d, want = 4", len(logits))
		}
		// logitは一致ビット数なので、0以上、総ビット数以下
		maxMatch := 1 * 32
		for i, logit := range logits {
			if logit < 0 || logit > maxMatch {
				t.Errorf("logits[%d]が範囲外: got = %d, want = [0, %d]", i, logit, maxMatch)
			}
		}
	})

	t.Run("性質_同じ入力に対して決定論的", func(t *testing.T) {
		logits1, err := model.PredictLogits(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		logits2, err := model.PredictLogits(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !slices.Equal(logits1, logits2) {
			t.Errorf("同じ入力に対して結果が異なる: %v != %v", logits1, logits2)
		}
	})

	t.Run("性質_softmaxの合計は1", func(t *testing.T) {
		probs, err := model.PredictSoftmax(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		var sum float32
		for i, p := range probs {
			if p < 0.0 || p > 1.0 {
				t.Errorf("probs[%d]が範囲外: got = %f", i, p)
			}
			sum += p
		}
		if math.Abs(float64(sum-1.0)) > 0.0001 {
			t.Errorf("合計の不一致: got = %f, want = 1.0(±0.0001)", sum)
		}
	})

	t.Run("性質_PredictValueはValuesの範囲内", func(t *testing.T) {
		val, err := model.PredictValue(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		minVal := model.Values[0]
		maxVal := model.Values[len(model.Values)-1]
		if val < minVal || val > maxVal {
			t.Errorf("値が範囲外: got = %f, want = [%f, %f]", val, minVal, maxVal)
		}
	})
}

func TestModelPredict_EmptyPrototypes(t *testing.T) {
	model := bep.Model{}
	for _, test := range []struct {
		name string
		call func() error
	}{
		{name: "PredictLogits", call: func() error { _, err := model.PredictLogits(nil); return err }},
		{name: "PredictSoftmax", call: func() error { _, err := model.PredictSoftmax(nil); return err }},
		{name: "PredictValue", call: func() error { _, err := model.PredictValue(nil); return err }},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.call(); err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
		})
	}
}

func TestModelSaveLoad(t *testing.T) {
	model, rng := newTestModel(t)
	x := newTestInput(t, rng)

	wantLogits, err := model.PredictLogits(x)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	path := filepath.Join(t.TempDir(), "model.gob")
	if err := model.Save(path); err != nil {
		t.Fatalf("保存失敗: %v", err)
	}

	loaded, err := bep.LoadModel(path)
	if err != nil {
		t.Fatalf("読み込み失敗: %v", err)
	}

	gotLogits, err := loaded.PredictLogits(x)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	// 保存前と読み込み後で、同じ入力に対する出力が一致する事
	if !slices.Equal(gotLogits, wantLogits) {
		t.Errorf("logitsの不一致: got = %v, want = %v", gotLogits, wantLogits)
	}
}

func TestModelSaveLoadKeepsLayerSettings(t *testing.T) {
	model, _ := newTestModel(t)
	d, ok := model.Backbone[0].(*bep.Dense)
	if !ok {
		t.Fatal("先頭の層が *bep.Dense ではない")
	}
	d.GroupSize = 3
	d.MaxAbsNoise = 5

	path := filepath.Join(t.TempDir(), "model.gob")
	if err := model.Save(path); err != nil {
		t.Fatalf("保存失敗: %v", err)
	}
	loaded, err := bep.LoadModel(path)
	if err != nil {
		t.Fatalf("読み込み失敗: %v", err)
	}

	got, ok := loaded.Backbone[0].(*bep.Dense)
	if !ok {
		t.Fatal("読み込んだ先頭の層が *bep.Dense ではない")
	}
	if got.GroupSize != 3 || got.MaxAbsNoise != 5 {
		t.Errorf("層の設定が保存されていない: got = (GroupSize %d, MaxAbsNoise %d), want = (3, 5)", got.GroupSize, got.MaxAbsNoise)
	}
	// 読み込んだ直後に、追加の設定なしで学習を始められる事
	if err := newTestTrainer(t, &loaded).Validate(); err != nil {
		t.Errorf("読み込んだモデルで Validate が失敗: %v", err)
	}
}

func TestModelLoss_Error(t *testing.T) {
	model, rng := newTestModel(t)
	if err := model.SetSigmoidValues(); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	x := newTestInput(t, rng)

	t.Run("異常_長さ不一致", func(t *testing.T) {
		_, err := model.Loss(bitsx.Matrices{x}, []int{0, 1}, 1)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})

	t.Run("異常_labelが範囲外", func(t *testing.T) {
		_, err := model.Loss(bitsx.Matrices{x}, []int{99}, 1)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "label") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})
}

func TestModelEvaluation_InvalidSize(t *testing.T) {
	model, rng := newTestModel(t)
	if err := model.SetSigmoidValues(); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	x := newTestInput(t, rng)

	for _, test := range []struct {
		name string
		call func() error
	}{
		{name: "Accuracy_入力が空", call: func() error { _, err := model.Accuracy(nil, nil, 1); return err }},
		{name: "Loss_入力が空", call: func() error { _, err := model.Loss(nil, nil, 1); return err }},
		{name: "Accuracy_ワーカー数0", call: func() error { _, err := model.Accuracy(bitsx.Matrices{x}, []int{0}, 0); return err }},
		{name: "Accuracy_ワーカー数が負", call: func() error { _, err := model.Accuracy(bitsx.Matrices{x}, []int{0}, -1); return err }},
		{name: "Loss_ワーカー数0", call: func() error { _, err := model.Loss(bitsx.Matrices{x}, []int{0}, 0); return err }},
		{name: "Loss_ワーカー数が負", call: func() error { _, err := model.Loss(bitsx.Matrices{x}, []int{0}, -1); return err }},
		{name: "PairwiseAccuracy_入力が空", call: func() error { _, err := model.PairwiseAccuracy(nil, 1); return err }},
		{name: "PairwiseAccuracy_ワーカー数0", call: func() error {
			_, err := model.PairwiseAccuracy(bep.RankPairXs{{High: x, Low: x}}, 0)
			return err
		}},
		{name: "PairwiseAccuracy_ワーカー数が負", call: func() error {
			_, err := model.PairwiseAccuracy(bep.RankPairXs{{High: x, Low: x}}, -1)
			return err
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.call(); err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
		})
	}
}

func TestModelPredictLabel(t *testing.T) {
	t.Run("正常_ロジットが最大のラベル", func(t *testing.T) {
		model, rng := newTestModel(t)
		for range 20 {
			x := newTestInput(t, rng)
			logits, err := model.PredictLogits(x)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			got, err := model.PredictLabel(x)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			// 同点のときは後ろのラベルを選ぶ
			for i, logit := range logits {
				if logit > logits[got] || (logit == logits[got] && i > got) {
					t.Fatalf("PredictLabel = %d だが、ラベル %d の方が優先されるべき: logits = %v", got, i, logits)
				}
			}
		}
	})

	t.Run("異常_Prototypes未設定", func(t *testing.T) {
		model, rng := newTestModel(t)
		x := newTestInput(t, rng)
		model.Prototypes = nil
		if _, err := model.PredictLabel(x); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}
