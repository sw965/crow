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
	x, err := bitsx.NewRandMatrix(1, 64, rng)
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

	// 必要マージン(一致数の差) = 2。旧APIの margin = 0.5, 総ビット数 = 8 に相当する。
	const marginBits = 2

	t.Run("正常_マージンを確保済みなら更新不要", func(t *testing.T) {
		// 正解proto0との距離0、proto1との距離8。差は8 >= 2 なので更新不要
		got, err := binary.SatisfiesUpdateCriterion(y, 0, prototypes, marginBits)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if got != false {
			t.Errorf("値の不一致: got = %t, want = false", got)
		}
	})

	t.Run("正常_マージン不足なら更新対象", func(t *testing.T) {
		// 正解proto1との距離8、proto0との距離0。差は-8 < 2 なので更新対象
		got, err := binary.SatisfiesUpdateCriterion(y, 1, prototypes, marginBits)
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
			if _, err := binary.SatisfiesUpdateCriterion(test.y, test.label, test.prototypes, marginBits); err == nil {
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

	for _, tt := range []struct {
		name        string
		maxAbsNoise func(cols int) int
		wantErr     bool
	}{
		{name: "異常_MaxAbsNoiseが負", maxAbsNoise: func(int) int { return -1 }, wantErr: true},
		{name: "正常_MaxAbsNoiseが0", maxAbsNoise: func(int) int { return 0 }},
		{name: "正常_MaxAbsNoiseが入力数", maxAbsNoise: func(cols int) int { return cols }},
		{name: "異常_MaxAbsNoiseが入力数を超える", maxAbsNoise: func(cols int) int { return cols + 1 }, wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model, _ := newTestModel(t)
			ctx := binary.NewSharedHyperparameters()
			if err := model.Backbone.SetSharedHyperparameters(&ctx); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			d, ok := model.Backbone[0].(*binary.Dense)
			if !ok {
				t.Fatal("先頭の層が *binary.Dense ではない")
			}
			d.MaxAbsNoise = tt.maxAbsNoise(d.W.Cols())
			err := binary.NewTrainer(model, 1).Validate()
			if tt.wantErr && err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("nilを期待したが、エラーが返された: %v", err)
			}
			if tt.wantErr && !strings.Contains(err.Error(), "MaxAbsNoise") {
				t.Errorf("エラーメッセージが不十分: %s", err.Error())
			}
		})
	}

	t.Run("異常_LRHalfPowが負", func(t *testing.T) {
		model, _ := newTestModel(t)
		ctx := binary.NewSharedHyperparameters()
		if err := model.Backbone.SetSharedHyperparameters(&ctx); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		trainer := binary.NewTrainer(model, 1)
		trainer.LRHalfPow = -1
		err := trainer.Validate()
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "LR") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})

	for _, margin := range []float32{-0.1, 1.1} {
		t.Run(fmt.Sprintf("異常_LogitMarginが範囲外_%g", margin), func(t *testing.T) {
			model, _ := newTestModel(t)
			ctx := binary.NewSharedHyperparameters()
			if err := model.Backbone.SetSharedHyperparameters(&ctx); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			trainer := binary.NewTrainer(model, 1)
			trainer.LogitMargin = margin
			err := trainer.Validate()
			if err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
			if !strings.Contains(err.Error(), "LogitMargin") {
				t.Errorf("エラーメッセージが不十分: %s", err.Error())
			}
		})
	}

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

// assertDenseConsistent は、W = sign(H) かつ WT = W の転置であることを確認する。
func assertDenseConsistent(t *testing.T, d *binary.Dense) {
	t.Helper()
	cols := d.W.Cols()
	for r := range d.W.Rows() {
		for c := range cols {
			bit, err := d.W.Bit(r, c)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			want := uint64(0)
			if d.H[r*cols+c] >= 0 {
				want = 1
			}
			if bit != want {
				t.Fatalf("W が sign(H) と不一致 (r = %d, c = %d): got = %d, want = %d", r, c, bit, want)
			}
		}
	}
	wt, err := d.W.Transpose()
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if !wt.Equal(d.WT) {
		t.Fatal("WT が W の転置と不一致")
	}
}

func TestDenseUpdate(t *testing.T) {
	clip := func(v int) int8 { return int8(max(math.MinInt8, min(v, math.MaxInt8))) }

	for _, shape := range [][2]int{{8, 64}, {16, 100}, {33, 37}} {
		t.Run(fmt.Sprintf("正常_lrHalfPowが0なら全要素を更新_%dx%d", shape[0], shape[1]), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(1, 2))
			d, err := binary.NewDense(shape[0], shape[1], rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			deltas := d.NewZerosDeltas()
			for i := range deltas[0] {
				// クリップと符号反転の両方を通すため、H の値域を超える大きさも混ぜる
				deltas[0][i] = int16(rng.IntN(401) - 200)
			}
			h0 := slices.Clone(d.H)

			if err := d.Update(deltas, 0, rng); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			for i, v := range d.H {
				if want := clip(int(h0[i]) + int(deltas[0][i])); v != want {
					t.Fatalf("H の不一致 (i = %d): got = %d, want = %d", i, v, want)
				}
			}
			assertDenseConsistent(t, d)
		})
	}

	for _, tt := range []struct {
		lrHalfPow int
		wantP     float64
	}{
		{lrHalfPow: 1, wantP: 0.5},
		{lrHalfPow: 3, wantP: 0.125},
	} {
		t.Run(fmt.Sprintf("正常_更新確率_lrHalfPow=%d", tt.lrHalfPow), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(3, 4))
			// 列数を64の倍数にしないことで、端数ワードの範囲外ビットを更新しないことも確認する
			d, err := binary.NewDense(64, 1000, rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			deltas := d.NewZerosDeltas()
			for i := range deltas[0] {
				// 0 を除き、更新されたかどうかを H の変化で判別できるようにする
				if rng.IntN(2) == 0 {
					deltas[0][i] = 1
				} else {
					deltas[0][i] = -1
				}
			}
			h0 := slices.Clone(d.H)

			if err := d.Update(deltas, tt.lrHalfPow, rng); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			updated := 0
			for i, v := range d.H {
				switch v {
				case h0[i]:
				case clip(int(h0[i]) + int(deltas[0][i])):
					updated++
				default:
					t.Fatalf("H が更新前とも更新後とも一致しない (i = %d): got = %d, h0 = %d, delta = %d", i, v, h0[i], deltas[0][i])
				}
			}
			// N=64000。tol=0.02 は p=0.5 で約10σ、p=0.125 で約15σ
			if gotP := float64(updated) / float64(len(d.H)); math.Abs(gotP-tt.wantP) > 0.02 {
				t.Errorf("更新確率の不一致: got = %f, want = %f", gotP, tt.wantP)
			}
			assertDenseConsistent(t, d)
		})
	}

	t.Run("異常_lrHalfPowが負", func(t *testing.T) {
		rng := rand.New(rand.NewPCG(5, 6))
		d, err := binary.NewDense(4, 10, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if err := d.Update(d.NewZerosDeltas(), -1, rng); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestDenseForwardNoise(t *testing.T) {
	const (
		rows  = 4
		xCols = 100
		wRows = 300
	)
	newDense := func(t *testing.T, maxAbsNoise int) (*binary.Dense, *bitsx.Matrix) {
		t.Helper()
		rng := rand.New(rand.NewPCG(11, 12))
		d, err := binary.NewDense(wRows, xCols, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		d.MaxAbsNoise = maxAbsNoise
		ctx := binary.NewSharedHyperparameters()
		if err := (binary.Sequence{d}).SetSharedHyperparameters(&ctx); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		x, err := bitsx.NewRandMatrix(rows, xCols, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		return d, x
	}

	t.Run("正常_MaxAbsNoiseが0ならPredictと一致し乱数を消費しない", func(t *testing.T) {
		d, x := newDense(t, 0)
		rng := rand.New(rand.NewPCG(1, 2))
		y, _, err := d.Forward(x, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		want, err := d.Predict(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !y.Equal(want) {
			t.Error("MaxAbsNoiseが0の Forward が Predict と一致しない")
		}
		// biaslab の差分テストは、ノイズ無効時に乱数を消費しないことを前提にしている
		if rng.Uint64() != rand.New(rand.NewPCG(1, 2)).Uint64() {
			t.Error("MaxAbsNoiseが0なのに乱数を消費した")
		}
	})

	t.Run("正常_MaxAbsNoiseを超える位置のニューロンは反転しない", func(t *testing.T) {
		const maxAbsNoise = 10
		d, x := newDense(t, maxAbsNoise)
		u, err := x.Dot(d.W)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		rng := rand.New(rand.NewPCG(3, 4))
		flipped := 0
		for range 20 {
			y, _, err := d.Forward(x, rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			for r := range rows {
				for c := range wRows {
					z := 2*u[r*wRows+c] - xCols
					bit, err := y.Bit(r, c)
					if err != nil {
						t.Fatalf("予期せぬエラー: %v", err)
					}
					noiseless := uint64(0)
					if z >= 0 {
						noiseless = 1
					}
					if bit != noiseless {
						if z > maxAbsNoise || z < -maxAbsNoise {
							t.Fatalf("|z| = %d が MaxAbsNoise %d を超えているのに反転した (r = %d, c = %d)", z, maxAbsNoise, r, c)
						}
						flipped++
					}
				}
			}
		}
		if flipped == 0 {
			t.Error("ノイズで反転したニューロンが1つも無い")
		}
	})
}

func TestSetNoiseScale(t *testing.T) {
	newSeq := func(t *testing.T) (binary.Sequence, *binary.Dense, *binary.Dense) {
		t.Helper()
		rng := rand.New(rand.NewPCG(21, 22))
		first, err := binary.NewDense(64, 784, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		second, err := binary.NewDense(10, 64, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		return binary.Sequence{first, second}, first, second
	}

	t.Run("正常_Denseの倍率3/4", func(t *testing.T) {
		_, first, _ := newSeq(t)
		if err := first.SetNoiseScale(3, 4); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if first.MaxAbsNoise != 36 {
			t.Errorf("MaxAbsNoise の不一致: got = %d, want = 36", first.MaxAbsNoise)
		}
	})

	t.Run("異常_Denseの倍率が不正なら値を変えない", func(t *testing.T) {
		_, first, _ := newSeq(t)
		before := first.MaxAbsNoise
		if err := first.SetNoiseScale(1, 0); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if first.MaxAbsNoise != before {
			t.Errorf("エラーなのに MaxAbsNoise が変わった: got = %d, want = %d", first.MaxAbsNoise, before)
		}
	})

	t.Run("正常_Sequenceで全層に設定", func(t *testing.T) {
		seq, first, second := newSeq(t)
		if err := seq.SetNoiseScale(1, 1); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		// isqrt(784) = 28、isqrt(64) = 8 に √3 ≈ 1.732 を掛けて切り捨て
		if first.MaxAbsNoise != 48 || second.MaxAbsNoise != 13 {
			t.Errorf("MaxAbsNoise の不一致: got = (%d, %d), want = (48, 13)", first.MaxAbsNoise, second.MaxAbsNoise)
		}
	})

	t.Run("異常_Sequenceで一部の層が範囲外なら全層を変えない", func(t *testing.T) {
		seq, first, second := newSeq(t)
		before1, before2 := first.MaxAbsNoise, second.MaxAbsNoise
		// 784入力の層は 484 で収まるが、64入力の層は 138 で入力数を超える
		err := seq.SetNoiseScale(10, 1)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "layer 1") {
			t.Errorf("エラーメッセージに層番号が無い: %s", err.Error())
		}
		if first.MaxAbsNoise != before1 || second.MaxAbsNoise != before2 {
			t.Errorf("エラーなのに値が変わった: got = (%d, %d), want = (%d, %d)",
				first.MaxAbsNoise, second.MaxAbsNoise, before1, before2)
		}
	})
}
