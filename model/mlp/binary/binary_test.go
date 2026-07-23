package binary_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/sw965/crow/model/mlp/binary"
	"github.com/sw965/omw/mathx/bitsx"
)

// newTestModel は、テスト用の小さなモデル(入力1x64, 隠れ層32, クラス数4)を返す。
// 乱数はシードを固定する(乱数が主目的の機能ではない為、テスト方針に従いシード固定で期待値・性質テストを行う)。
func newTestModel(t *testing.T) (binary.Model, *rand.Rand) {
	t.Helper()
	rng := rand.New(rand.NewPCG(1, 2))

	model := binary.Model{XRows: 1, XCols: 64}
	if err := model.AppendDenseLayer(32, rng); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if err := model.SetClassPrototypes(4, rng); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	return model, rng
}

func newTestInput(t *testing.T, rng *rand.Rand) *bitsx.Matrix {
	t.Helper()
	x, err := bitsx.NewRandMatrix(1, 64, 0, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	return x
}

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
		model := binary.Model{}
		err := model.SetValues(0.0, 1.0)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestModelValueToLabel(t *testing.T) {
	model := binary.Model{Values: []float32{0.0, 0.25, 0.5, 0.75, 1.0}}

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
		empty := binary.Model{}
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
		m := binary.Model{Values: values}

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

		for i := 0; i < 1000; i++ {
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
	model := binary.Model{}
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

	loaded, err := binary.LoadModel(path)
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

func TestSatisfiesUpdateCriterion(t *testing.T) {
	newMatrix := func(setBits []int) *bitsx.Matrix {
		m, err := bitsx.NewZerosMatrix(1, 8)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		for _, c := range setBits {
			if err := m.Set(0, c); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
		}
		return m
	}

	y := newMatrix([]int{0, 1, 2, 3})
	// proto0はyと完全一致、proto1はyと完全不一致
	proto0 := newMatrix([]int{0, 1, 2, 3})
	proto1 := newMatrix([]int{4, 5, 6, 7})
	prototypes := bitsx.Matrices{proto0, proto1}

	// margin = 0.5, 総ビット数 = 8 の場合、必要マージン(一致数の差) = 8 * 0.5 / 2 = 2

	t.Run("正常_マージンを確保済みなら更新不要", func(t *testing.T) {
		// 正解proto0との距離0、proto1との距離8。差は8 >= 2 なので更新不要
		got, err := binary.SatisfiesUpdateCriterion(y, 0, prototypes, 0.5)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if got != false {
			t.Errorf("値の不一致: got = %t, want = false", got)
		}
	})

	t.Run("正常_マージン不足なら更新対象", func(t *testing.T) {
		// 正解proto1との距離8、proto0との距離0。差は-8 < 2 なので更新対象
		got, err := binary.SatisfiesUpdateCriterion(y, 1, prototypes, 0.5)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if got != true {
			t.Errorf("値の不一致: got = %t, want = true", got)
		}
	})

	for _, test := range []struct {
		name       string
		y          *bitsx.Matrix
		label      int
		prototypes bitsx.Matrices
	}{
		{name: "異常_出力がnil", label: 0, prototypes: prototypes},
		{name: "異常_Prototypesが空", y: y, label: 0},
		{name: "異常_labelが負", y: y, label: -1, prototypes: prototypes},
		{name: "異常_labelが上限以上", y: y, label: len(prototypes), prototypes: prototypes},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := binary.SatisfiesUpdateCriterion(test.y, test.label, test.prototypes, 0.5); err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
		})
	}
}

func TestDelta(t *testing.T) {
	t.Run("正常_Add", func(t *testing.T) {
		d := binary.Delta{1, -2, 3}
		if err := d.Add(binary.Delta{10, 20, 30}); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		want := binary.Delta{11, 18, 33}
		if !slices.Equal(d, want) {
			t.Errorf("値の不一致: got = %v, want = %v", d, want)
		}
	})

	t.Run("異常_Add_長さ不一致", func(t *testing.T) {
		d := binary.Delta{1, 2}
		err := d.Add(binary.Delta{1})
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})

	t.Run("正常_Sign", func(t *testing.T) {
		d := binary.Delta{5, -3, 0, 100, -1}
		d.Sign()
		want := binary.Delta{1, -1, 0, 1, -1}
		if !slices.Equal(d, want) {
			t.Errorf("値の不一致: got = %v, want = %v", d, want)
		}
	})

	t.Run("正常_Aggregate", func(t *testing.T) {
		sd1 := binary.SeqDelta{binary.Deltas{binary.Delta{1, 2}}}
		sd2 := binary.SeqDelta{binary.Deltas{binary.Delta{10, 20}}}
		dst := binary.SeqDelta{binary.Deltas{binary.Delta{99, 99}}} // Aggregate前にクリアされる事も確認

		sds := binary.SeqDeltas{sd1, sd2}
		if err := sds.Aggregate(dst); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		want := binary.Delta{11, 22}
		if !slices.Equal(dst[0][0], want) {
			t.Errorf("値の不一致: got = %v, want = %v", dst[0][0], want)
		}
	})

	t.Run("異常_Aggregate_空", func(t *testing.T) {
		sds := binary.SeqDeltas{}
		err := sds.Aggregate(binary.SeqDelta{})
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestTrainerValidate(t *testing.T) {
	t.Run("異常_Backboneが空", func(t *testing.T) {
		model := binary.Model{}
		trainer := binary.NewTrainer(model, 1)
		err := trainer.Validate()
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "Backbone") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})

	t.Run("異常_sharedHyperparameters未設定", func(t *testing.T) {
		model, _ := newTestModel(t)
		trainer := binary.NewTrainer(model, 1)
		err := trainer.Validate()
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "sharedHyperparameters") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})

	t.Run("異常_LRが0以下", func(t *testing.T) {
		model, _ := newTestModel(t)
		ctx := binary.NewSharedHyperparameters()
		if err := model.Backbone.SetSharedHyperparameters(&ctx); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		trainer := binary.NewTrainer(model, 1)
		trainer.LR = 0.0
		err := trainer.Validate()
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "LR") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})

	t.Run("異常_Valuesが昇順ではない", func(t *testing.T) {
		model, _ := newTestModel(t)
		ctx := binary.NewSharedHyperparameters()
		if err := model.Backbone.SetSharedHyperparameters(&ctx); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		model.Values = []float32{1.0, 0.0}
		trainer := binary.NewTrainer(model, 1)
		err := trainer.Validate()
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "昇順") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})

	t.Run("正常", func(t *testing.T) {
		model, _ := newTestModel(t)
		ctx := binary.NewSharedHyperparameters()
		if err := model.Backbone.SetSharedHyperparameters(&ctx); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		trainer := binary.NewTrainer(model, 1)
		if err := trainer.Validate(); err != nil {
			t.Errorf("予期せぬエラー: %v", err)
		}
	})

	for _, p := range []int{0, -1} {
		t.Run(fmt.Sprintf("異常_ワーカー数_%d", p), func(t *testing.T) {
			trainer := binary.NewTrainer(binary.Model{}, p)
			if err := trainer.Validate(); err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
		})
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
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.call(); err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
		})
	}
}
