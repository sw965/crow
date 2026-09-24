package bep_test

import (
	"math/rand/v2"
	"testing"

	"github.com/sw965/crow/model/bep"
	"github.com/sw965/omw/mathx/bitsx"
)

// newTestModel は、テスト用の小さなモデル(入力1x64, 隠れ層32, クラス数4)を返す。
// 乱数はシードを固定する(乱数が主目的の機能ではない為、テスト方針に従いシード固定で期待値・性質テストを行う)。
func newTestModel(t *testing.T) (bep.Model, *rand.Rand) {
	t.Helper()
	rng := rand.New(rand.NewPCG(1, 2))

	model := bep.Model{XRows: 1, XCols: 64}
	if err := model.AppendDenseLayer(32, rng); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if err := model.SetClassPrototypes(4, rng); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	return model, rng
}

func newTestTrainer(t *testing.T, model *bep.Model) *bep.Trainer {
	t.Helper()
	trainer, err := bep.NewTrainer(model, 1)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	return trainer
}

func newTestInput(t *testing.T, rng *rand.Rand) *bitsx.Matrix {
	t.Helper()
	x, err := bitsx.NewRandMatrix(1, 64, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	return x
}
