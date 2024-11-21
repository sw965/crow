package model

import (
	"fmt"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
)

type GradComputer[M any, Fs ~[]F, Ls ~[]L, F, L, G any] func(M, Fs, Ls) (G, error)

type Optimizer[M any, G any] func(M, G, float64) error

type Trainer[M any, Fs ~[]F, Ls ~[]L, F, L, G any] struct {
	Features Fs
	Labels   Ls
	GradComputer GradComputer[M, Fs, Ls, F, L, G]
	Optimizer Optimizer[M, G]
	BatchSize int
	Epoch int
}

func (t *Trainer[M, Xs, Ts, X, T, G]) Train(model M, lr float64, r *rand.Rand) error {
	if t.Optimizer == nil {
		return fmt.Errorf("Optimizerが設定されていない為、モデルの訓練を開始できません。")
	}

	n := len(t.Features)
	if n < t.BatchSize {
		return fmt.Errorf("データ数 < バッチサイズである為、モデルの訓練を出来ません、")
	}

	if t.Epoch <= 0 {
		return fmt.Errorf("エポック数が0以下である為、モデルの訓練を開始出来ません。")
	}

	iter := n / t.BatchSize * t.Epoch
	for i := 0; i < iter; i++ {
		idxs := omwrand.Ints(t.BatchSize, 0, n, r)
		xs := omwslices.ElementsByIndices(t.Features, idxs...)
		ts := omwslices.ElementsByIndices(t.Labels, idxs...)
		grad, err := t.GradComputer(model, xs, ts)
		if err != nil {
			return err
		}
		err = t.Optimizer(model, grad, lr)
		if err != nil {
			return err
		}
	}
	return nil
}