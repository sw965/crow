package binary

import (
	"bytes"
	"cmp"
	"encoding/gob"
	"fmt"
	"math/rand/v2"
	"slices"

	"github.com/sw965/omw/mathx/bitsx"
)

// Clone はモデルの深いコピーを返す。
// gob経由でコピーするため、各層の sharedHyperparameters は引き継がれない。
// コピー先で学習する場合は Backbone.SetSharedHyperparameters を呼び直すこと。
func (m *Model) Clone() (Model, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(m); err != nil {
		return Model{}, err
	}
	var c Model
	if err := gob.NewDecoder(&buf).Decode(&c); err != nil {
		return Model{}, err
	}
	return c, nil
}

// SplitTrainValidation は xs をシャッフルして train / validation に分割する。
// ハイパラ選択や停滞判断に test データを使ってしまう(テストリーク)のを防ぐためのヘルパー。
func SplitTrainValidation(xs bitsx.Matrices, labels []int, valRatio float64, rng *rand.Rand) (
	trainXs bitsx.Matrices, trainLabels []int, valXs bitsx.Matrices, valLabels []int, err error) {

	n := len(xs)
	if n != len(labels) {
		return nil, nil, nil, nil, fmt.Errorf("長さが不一致: len(xs) = %d, len(labels) = %d", n, len(labels))
	}

	valN := int(float64(n) * valRatio)
	if valN <= 0 || valN >= n {
		return nil, nil, nil, nil, fmt.Errorf("valRatio %g では分割できません (n=%d, val=%d)", valRatio, n, valN)
	}

	perm := rng.Perm(n)
	valXs = make(bitsx.Matrices, 0, valN)
	valLabels = make([]int, 0, valN)
	trainXs = make(bitsx.Matrices, 0, n-valN)
	trainLabels = make([]int, 0, n-valN)

	for i, idx := range perm {
		if i < valN {
			valXs = append(valXs, xs[idx])
			valLabels = append(valLabels, labels[idx])
		} else {
			trainXs = append(trainXs, xs[idx])
			trainLabels = append(trainLabels, labels[idx])
		}
	}
	return trainXs, trainLabels, valXs, valLabels, nil
}

// SearchSpace はグリッドサーチの探索空間。
// nil (または空) のフィールドは「現行デフォルト値の1点」として扱う。
type SearchSpace struct {
	LRs                     []float32
	Margins                 []float32 // 論文(BEP)の r スケール
	GroupSizes              []int
	NoiseStdScales          []float32
	GateDropThresholdScales []float32
}

func (s SearchSpace) withDefaults() SearchSpace {
	ctx := NewSharedHyperparameters()
	if len(s.LRs) == 0 {
		s.LRs = []float32{defaultLR}
	}
	if len(s.Margins) == 0 {
		s.Margins = []float32{defaultMargin}
	}
	if len(s.GroupSizes) == 0 {
		s.GroupSizes = []int{ctx.GroupSize}
	}
	if len(s.NoiseStdScales) == 0 {
		s.NoiseStdScales = []float32{ctx.NoiseStdScale}
	}
	if len(s.GateDropThresholdScales) == 0 {
		s.GateDropThresholdScales = []float32{ctx.GateDropThresholdScale}
	}
	return s
}

// combinations は全フィールドの直積を列挙する。
func (s SearchSpace) combinations() []SearchParams {
	var params []SearchParams
	for _, lr := range s.LRs {
		for _, margin := range s.Margins {
			for _, gsize := range s.GroupSizes {
				for _, noise := range s.NoiseStdScales {
					for _, gate := range s.GateDropThresholdScales {
						params = append(params, SearchParams{
							LR:                     lr,
							Margin:                 margin,
							GroupSize:              gsize,
							NoiseStdScale:          noise,
							GateDropThresholdScale: gate,
						})
					}
				}
			}
		}
	}
	return params
}

// SearchParams は1試行分のハイパーパラメータの組。
type SearchParams struct {
	LR                     float32
	Margin                 float32
	GroupSize              int
	NoiseStdScale          float32
	GateDropThresholdScale float32
}

func (p SearchParams) String() string {
	return fmt.Sprintf("lr=%g margin=%g gsize=%d noise=%g gate=%g",
		p.LR, p.Margin, p.GroupSize, p.NoiseStdScale, p.GateDropThresholdScale)
}

// TrialResult は1試行の結果。
type TrialResult struct {
	Params     SearchParams
	ValAccs    []float32 // 各エポック終了時のvalidation精度
	BestValAcc float32
	BestEpoch  int
}

// GridSearch は探索空間の全組み合わせについて、base のコピーを epochs エポック学習し、
// validation精度で評価する。結果は BestValAcc の降順で返す。
//
//   - 全試行は同一の初期モデル(base の深いコピー)から開始するため、公平に比較できる。
//     base 自体は変更されない。
//   - テストリークを防ぐため、testデータはこの関数に渡さないこと(分割は SplitTrainValidation 参照)。
//   - logf に t.Logf や fmt.Printf 相当を渡すと1試行ごとに進捗を出力する。nil なら無音。
func GridSearch(base Model, trainXs bitsx.Matrices, trainLabels []int, valXs bitsx.Matrices, valLabels []int,
	epochs, miniBatchSize, p int, space SearchSpace, logf func(format string, a ...any)) ([]TrialResult, error) {

	if epochs <= 0 {
		return nil, fmt.Errorf("epochs <= 0: epochs > 0 であるべき")
	}
	if logf == nil {
		logf = func(string, ...any) {}
	}

	space = space.withDefaults()
	combos := space.combinations()
	for _, prm := range combos {
		if prm.LR <= 0 || prm.GroupSize < 1 {
			return nil, fmt.Errorf("不正なパラメータ: %s", prm)
		}
	}

	results := make([]TrialResult, 0, len(combos))
	for i, prm := range combos {
		result, err := runSearchTrial(&base, prm, trainXs, trainLabels, valXs, valLabels, epochs, miniBatchSize, p)
		if err != nil {
			return nil, fmt.Errorf("trial %s: %w", prm, err)
		}
		logf("[%d/%d] %s -> best val acc %.4f (epoch %d)", i+1, len(combos), prm, result.BestValAcc, result.BestEpoch)
		results = append(results, result)
	}

	slices.SortStableFunc(results, func(a, b TrialResult) int {
		return cmp.Compare(b.BestValAcc, a.BestValAcc)
	})
	return results, nil
}

func runSearchTrial(base *Model, prm SearchParams, trainXs bitsx.Matrices, trainLabels []int,
	valXs bitsx.Matrices, valLabels []int, epochs, miniBatchSize, p int) (TrialResult, error) {

	model, err := base.Clone()
	if err != nil {
		return TrialResult{}, err
	}

	ctx := NewSharedHyperparameters()
	ctx.GroupSize = prm.GroupSize
	ctx.NoiseStdScale = prm.NoiseStdScale
	ctx.GateDropThresholdScale = prm.GateDropThresholdScale
	if err := model.Backbone.SetSharedHyperparameters(&ctx); err != nil {
		return TrialResult{}, err
	}

	trainer := NewTrainer(model, p)
	trainer.MiniBatchSize = miniBatchSize
	trainer.LR = prm.LR
	trainer.Margin = prm.Margin

	result := TrialResult{
		Params:    prm,
		ValAccs:   make([]float32, 0, epochs),
		BestEpoch: -1,
	}

	for e := 0; e < epochs; e++ {
		if err := trainer.Train(trainXs, trainLabels); err != nil {
			return TrialResult{}, err
		}

		acc, err := model.Accuracy(valXs, valLabels, p)
		if err != nil {
			return TrialResult{}, err
		}

		result.ValAccs = append(result.ValAccs, acc)
		if result.BestEpoch < 0 || acc > result.BestValAcc {
			result.BestValAcc = acc
			result.BestEpoch = e
		}
	}
	return result, nil
}
