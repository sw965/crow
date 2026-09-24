package bep_test

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/sw965/crow/model/bep"
	"github.com/sw965/omw/mathx/bitsx"
)

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
		got, err := bep.SatisfiesUpdateCriterion(y, 0, prototypes, marginBits)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if got != false {
			t.Errorf("値の不一致: got = %t, want = false", got)
		}
	})

	t.Run("正常_マージン不足なら更新対象", func(t *testing.T) {
		// 正解proto1との距離8、proto0との距離0。差は-8 < 2 なので更新対象
		got, err := bep.SatisfiesUpdateCriterion(y, 1, prototypes, marginBits)
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
			if _, err := bep.SatisfiesUpdateCriterion(test.y, test.label, test.prototypes, marginBits); err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
		})
	}
}

func TestTrainerValidate(t *testing.T) {
	t.Run("異常_NewTrainerを通さないゼロ値", func(t *testing.T) {
		var trainer bep.Trainer
		err := trainer.Validate()
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "model") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})

	t.Run("異常_Backboneが空", func(t *testing.T) {
		model := bep.Model{}
		trainer := newTestTrainer(t, &model)
		err := trainer.Validate()
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "Backbone") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})

	t.Run("異常_GroupSizeが0以下", func(t *testing.T) {
		model, _ := newTestModel(t)
		d, ok := model.Backbone[0].(*bep.Dense)
		if !ok {
			t.Fatal("先頭の層が *bep.Dense ではない")
		}
		d.GroupSize = 0
		err := newTestTrainer(t, &model).Validate()
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "GroupSize") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})

	for _, tt := range []struct {
		name    string
		setBias func(d *bep.Dense)
	}{
		{name: "異常_Biasの長さが出力数と違う", setBias: func(d *bep.Dense) { d.Bias = d.Bias[:1] }},
		{name: "異常_Biasが入力数を超える", setBias: func(d *bep.Dense) { d.Bias[0] = int32(d.W.Cols() + 1) }},
		{name: "異常_Biasが負の入力数を下回る", setBias: func(d *bep.Dense) { d.Bias[0] = -int32(d.W.Cols() + 1) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model, _ := newTestModel(t)
			d := model.Backbone[0].(*bep.Dense)
			tt.setBias(d)
			err := newTestTrainer(t, &model).Validate()
			if err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
			if !strings.Contains(err.Error(), "Bias") {
				t.Errorf("エラーメッセージが不十分: %s", err.Error())
			}
		})
	}

	t.Run("正常_Biasが入力数ちょうど", func(t *testing.T) {
		model, _ := newTestModel(t)
		d := model.Backbone[0].(*bep.Dense)
		d.Bias[0] = int32(d.W.Cols())
		d.Bias[1] = -int32(d.W.Cols())
		if err := newTestTrainer(t, &model).Validate(); err != nil {
			t.Errorf("予期せぬエラー: %v", err)
		}
	})

	t.Run("異常_MiniBatchSizeが0以下", func(t *testing.T) {
		model, _ := newTestModel(t)
		trainer := newTestTrainer(t, &model)
		trainer.MiniBatchSize = 0
		err := trainer.Validate()
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "MiniBatchSize") {
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
			d, ok := model.Backbone[0].(*bep.Dense)
			if !ok {
				t.Fatal("先頭の層が *bep.Dense ではない")
			}
			d.MaxAbsNoise = tt.maxAbsNoise(d.W.Cols())
			err := newTestTrainer(t, &model).Validate()
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
		trainer := newTestTrainer(t, &model)
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
			trainer := newTestTrainer(t, &model)
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

	t.Run("異常_Valuesがあるのに温度計ではない", func(t *testing.T) {
		model, _ := newTestModel(t)
		if err := model.SetSigmoidValues(); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		err := newTestTrainer(t, &model).Validate()
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "温度計") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})

	t.Run("正常_温度計の回帰モデル", func(t *testing.T) {
		model, _ := newTestModel(t)
		if err := model.SetRegressionPrototypes(11); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if err := model.SetSigmoidValues(); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if err := newTestTrainer(t, &model).Validate(); err != nil {
			t.Errorf("予期せぬエラー: %v", err)
		}
	})

	t.Run("異常_Valuesが昇順ではない", func(t *testing.T) {
		model, _ := newTestModel(t)
		model.Values = []float32{1.0, 0.0}
		trainer := newTestTrainer(t, &model)
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
		trainer := newTestTrainer(t, &model)
		if err := trainer.Validate(); err != nil {
			t.Errorf("予期せぬエラー: %v", err)
		}
	})

	t.Run("異常_modelがnil", func(t *testing.T) {
		if _, err := bep.NewTrainer(nil, 1); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})

	t.Run("正常_作成後のモデルの変更が反映される", func(t *testing.T) {
		rng := rand.New(rand.NewPCG(1, 2))
		model := bep.Model{XRows: 1, XCols: 64}
		if err := model.AppendDenseLayer(32, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		trainer := newTestTrainer(t, &model)
		if err := trainer.Validate(); err == nil {
			t.Fatal("Prototypes 未設定なのにエラーにならない")
		}
		// Trainer がモデルを参照しているので、後から設定したプロトタイプも見える
		if err := model.SetClassPrototypes(4, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if err := trainer.Validate(); err != nil {
			t.Errorf("後から設定したプロトタイプが反映されていない: %v", err)
		}
	})

	for _, p := range []int{0, -1} {
		t.Run(fmt.Sprintf("異常_ワーカー数_%d", p), func(t *testing.T) {
			if _, err := bep.NewTrainer(&bep.Model{}, p); err == nil {
				t.Fatal("エラーを期待したが、nilが返された")
			}
		})
	}
}
