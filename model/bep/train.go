package bep

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
)

const (
	// defaultLRHalfPow は、更新確率 (1/2)^3 = 1/8。旧既定値 LR = 0.1 と比べ、
	// MNIST / Fashion-MNIST の3シードで精度が誤差の範囲で変わらなかった。
	defaultLRHalfPow = 3
	// defaultLogitMargin は、10クラスのETF(出力1024ビット)で旧方式(総ビット数/2 基準の 0.5 = 256ビット)と
	// ほぼ同じ厳しさになる値。0.5 のままだと約281ビットになり、3シードの実測で精度がわずかに下がった。
	defaultLogitMargin    float32 = 0.45
	defaultPairwiseMargin float32 = 0.5
)

type Trainer struct {
	MiniBatchSize int
	LRHalfPow     int

	// LogitMargin は更新判定のマージンで、プロトタイプ間の最小距離に対する比率(0.0〜1.0)。
	// 出力の総ビット数ではなく最小距離を基準にするのは、達成しうるリードの上限が
	// 最小距離で決まるため。総ビット数を基準にすると、隣接プロトタイプが近い順序符号
	// (回帰の温度計)では小さな値でも上限を超え、判定が常に true になる。
	// 0 だと同点ロジットが更新対象にならないため、正の値を推奨。
	LogitMargin float32

	// PairwiseMargin は TrainPairwise が使う、値域に対する比率(0.0〜1.0)。
	// 復号した値の差を測るため、LogitMargin とは単位が異なる。
	PairwiseMargin float32

	model           *Model
	workerRNGs      []*rand.Rand
	shuffleRNG      *rand.Rand
	workerDeltas    SeqDeltas
	aggregatedDelta SeqDelta
}

func NewTrainer(model *Model, p int) (*Trainer, error) {
	if model == nil {
		return nil, errors.New("modelがnilです")
	}
	if p <= 0 {
		return nil, fmt.Errorf("ワーカー数が不正: p = %d: p > 0 であるべき", p)
	}
	workerCount := p
	workerDeltas := make(SeqDeltas, workerCount)
	backbone := model.Backbone
	numLayers := len(backbone)

	// ワーカーごとのバッファの初期化
	for i := range workerCount {
		sd := make(SeqDelta, numLayers)
		for l, layer := range backbone {
			sd[l] = layer.NewZerosDeltas()
		}
		workerDeltas[i] = sd
	}

	// 集約用バッファの初期化
	aggregatedDelta := make(SeqDelta, numLayers)
	for l, layer := range backbone {
		aggregatedDelta[l] = layer.NewZerosDeltas()
	}

	workerRNGs, err := randx.NewPCGs(workerCount)
	if err != nil {
		return nil, err
	}

	return &Trainer{
		MiniBatchSize:   128,
		LRHalfPow:       defaultLRHalfPow,
		LogitMargin:     defaultLogitMargin,
		PairwiseMargin:  defaultPairwiseMargin,
		model:           model,
		workerRNGs:      workerRNGs,
		shuffleRNG:      randx.NewPCG(),
		workerDeltas:    workerDeltas,
		aggregatedDelta: aggregatedDelta,
	}, nil
}

// minPrototypeHammingDistance は、プロトタイプ同士のハミング距離の最小値を返す。
// プロトタイプが2個未満のときは比べる相手が無いため 0 を返す。
func minPrototypeHammingDistance(prototypes bitsx.Matrices) (int, error) {
	if len(prototypes) < 2 {
		return 0, nil
	}
	minDist := math.MaxInt
	for i := range prototypes {
		for j := i + 1; j < len(prototypes); j++ {
			d, err := prototypes[i].HammingDistance(prototypes[j])
			if err != nil {
				return 0, err
			}
			minDist = min(minDist, d)
		}
	}
	return minDist, nil
}

func (t *Trainer) marginBits() (int, error) {
	minDist, err := minPrototypeHammingDistance(t.model.Prototypes)
	if err != nil {
		return 0, err
	}
	return int(t.LogitMargin * float32(minDist)), nil
}

func (t *Trainer) Train(xs bitsx.Matrices, labels []int) error {
	if err := t.Validate(); err != nil {
		return err
	}

	// 学習中はプロトタイプが変わらないので、バッチごとではなく呼び出しごとに1回だけ求める
	marginBits, err := t.marginBits()
	if err != nil {
		return err
	}

	n := len(xs)
	batchSize := min(t.MiniBatchSize, n)

	shuffledIdxs := t.shuffleRNG.Perm(n)
	for i := 0; i < n; i += batchSize {
		end := min(i+batchSize, n)

		batchIdxs := shuffledIdxs[i:end]
		batchXs, err := slicesx.ElementsByIndices(xs, batchIdxs...)
		if err != nil {
			return err
		}

		batchLabels, err := slicesx.ElementsByIndices(labels, batchIdxs...)
		if err != nil {
			return err
		}

		seqDelta, err := t.computeSeqSignDelta(batchXs, batchLabels, marginBits)
		if err != nil {
			return err
		}

		err = t.model.Backbone.Update(seqDelta, t.LRHalfPow, t.workerRNGs)
		if err != nil {
			return err
		}
	}
	return nil
}

func (t *Trainer) ComputeSeqSignDelta(xs bitsx.Matrices, labels []int) (SeqDelta, error) {
	marginBits, err := t.marginBits()
	if err != nil {
		return nil, err
	}
	return t.computeSeqSignDelta(xs, labels, marginBits)
}

func (t *Trainer) computeSeqSignDelta(xs bitsx.Matrices, labels []int, marginBits int) (SeqDelta, error) {
	n := len(xs)
	if n > math.MaxInt16 {
		return nil, fmt.Errorf("サンプル数が多すぎます: n = %d: Deltaの要素はint16の為、%d 以下であるべき", n, math.MaxInt16)
	}

	if n != len(labels) {
		return nil, fmt.Errorf("長さが不一致: len(xs) = %d, len(labels) = %d", n, len(labels))
	}

	p := len(t.workerRNGs)
	t.workerDeltas.Clear()
	backbone := t.model.Backbone
	prototypes := t.model.Prototypes

	err := parallel.For(n, p, func(workerID, idx int) error {
		rng := t.workerRNGs[workerID]
		x := xs[idx]
		label := labels[idx]

		y, backwards, err := backbone.Forward(x, rng)
		if err != nil {
			return err
		}

		shouldUpdate, err := SatisfiesUpdateCriterion(y, label, prototypes, marginBits)
		if err != nil {
			return err
		}

		if !shouldUpdate {
			return nil
		}

		target := prototypes[label]
		_, err = backwards.Propagate(target, t.workerDeltas[workerID])
		if err != nil {
			return err
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	err = t.workerDeltas.Aggregate(t.aggregatedDelta)
	if err != nil {
		return nil, err
	}

	t.aggregatedDelta.Sign()
	return t.aggregatedDelta, nil
}

func (t *Trainer) Validate() error {
	if t.model == nil {
		return errors.New("modelがnilです: NewTrainerで作成するべき")
	}

	if len(t.model.Backbone) == 0 {
		return errors.New("model validation: Backboneが空です: 学習前に1層以上追加するべき")
	}

	if len(t.model.Prototypes) == 0 {
		return errors.New("prototypesが未設定です: 学習前にSetClassPrototypes等で設定するべき")
	}

	if t.MiniBatchSize <= 0 {
		return fmt.Errorf("MiniBatchSizeが不正(MiniBatchSize <= 0): MiniBatchSize = %d: 1以上であるべき", t.MiniBatchSize)
	}

	if t.LogitMargin < 0.0 || t.LogitMargin > 1.0 {
		return fmt.Errorf("LogitMarginが不正: LogitMargin = %g: 0.0 <= LogitMargin <= 1.0 であるべき", t.LogitMargin)
	}

	if t.LRHalfPow < 0 {
		return fmt.Errorf("LRHalfPowが不正(LRHalfPow < 0): LRHalfPow = %d: LRHalfPow >= 0 であるべき", t.LRHalfPow)
	}

	if len(t.workerRNGs) == 0 {
		return errors.New("workerRNGsが空です: NewTrainerのpは1以上であるべき")
	}

	if err := t.model.validateAscendingValues(); err != nil {
		return err
	}

	for i, layer := range t.model.Backbone {
		d, ok := layer.(*Dense)
		if !ok {
			continue
		}
		// 0 以下だと逆伝播の updateK の計算で 0 除算になるため
		if d.GroupSize < 1 {
			return fmt.Errorf("layer %d: GroupSizeが不正: GroupSize = %d: 1以上であるべき", i, d.GroupSize)
		}
		// cols を超えると最も確信の強いニューロンまで反転しうるようになり、出力が無作為に近づくだけなので弾く
		if d.MaxAbsNoise < 0 || d.MaxAbsNoise > d.W.Cols() {
			return fmt.Errorf("layer %d: MaxAbsNoiseが不正: MaxAbsNoise = %d: 0 <= MaxAbsNoise <= %d (入力数) であるべき", i, d.MaxAbsNoise, d.W.Cols())
		}
	}

	return nil
}

func SatisfiesUpdateCriterion(y *bitsx.Matrix, label int, prototypes bitsx.Matrices, marginBits int) (bool, error) {
	if y == nil {
		return false, errors.New("yがnilです")
	}
	if len(prototypes) == 0 {
		return false, errors.New("prototypesが空です")
	}
	if label < 0 || label >= len(prototypes) {
		return false, fmt.Errorf("labelが範囲外: label = %d: 0 <= label < %d であるべき", label, len(prototypes))
	}

	t := prototypes[label]
	yMismatch, err := y.HammingDistance(t)
	if err != nil {
		return false, err
	}

	for i, proto := range prototypes {
		if i == label {
			continue
		}

		mismatch, err := y.HammingDistance(proto)
		if err != nil {
			return false, err
		}

		// 設定したマージンよりも差を付けられなかった場合、学習対象
		if (mismatch - yMismatch) < marginBits {
			return true, nil
		}
	}
	return false, nil
}
