package binary

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
)

const (
	defaultLR     float32 = 0.1
	defaultMargin float32 = 0.5
)

type Trainer struct {
	MiniBatchSize int
	LR            float32
	// Margin は論文(BEP)の r に相当(±1内積スケール、推奨 r ∈ (0,1])。
	// 0 だと同点ロジットが更新対象にならないため、正の値を推奨。
	Margin float32

	model           Model
	workerRNGs      []*rand.Rand
	shuffleRNG      *rand.Rand
	workerDeltas    SeqDeltas
	aggregatedDelta SeqDelta
}

func NewTrainer(model Model, p int) *Trainer {
	workerCount := max(p, 0)
	workerDeltas := make(SeqDeltas, workerCount)
	backbone := model.Backbone
	numLayers := len(backbone)

	// ワーカーごとのバッファの初期化
	for i := 0; i < workerCount; i++ {
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

	return &Trainer{
		MiniBatchSize:   128,
		LR:              defaultLR,
		Margin:          defaultMargin,
		model:           model,
		workerRNGs:      randx.NewPCGs(workerCount),
		shuffleRNG:      randx.NewPCG(),
		workerDeltas:    workerDeltas,
		aggregatedDelta: aggregatedDelta,
	}
}

func (t *Trainer) Train(xs bitsx.Matrices, labels []int) error {
	if err := t.Validate(); err != nil {
		return err
	}

	batchSize := t.MiniBatchSize
	if batchSize <= 0 {
		return fmt.Errorf("MiniBatchSizeが不正(MiniBatchSize <= 0): MiniBatchSize = %d: 1以上であるべき", batchSize)
	}

	n := len(xs)
	if n < batchSize {
		batchSize = n
	}

	shuffledIdxs := t.shuffleRNG.Perm(n)
	for i := 0; i < n; i += batchSize {
		end := i + batchSize
		if end > n {
			end = n
		}

		batchIdxs := shuffledIdxs[i:end]
		batchXs, err := slicesx.ElementsByIndices(xs, batchIdxs...)
		if err != nil {
			return err
		}

		batchLabels, err := slicesx.ElementsByIndices(labels, batchIdxs...)
		if err != nil {
			return err
		}

		seqDelta, err := t.ComputeSeqSignDelta(batchXs, batchLabels)
		if err != nil {
			return err
		}

		err = t.model.Backbone.Update(seqDelta, t.LR, t.workerRNGs)
		if err != nil {
			return err
		}
	}
	return nil
}

func (t *Trainer) ComputeSeqSignDelta(xs bitsx.Matrices, labels []int) (SeqDelta, error) {
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

		shouldUpdate, err := SatisfiesUpdateCriterion(y, label, prototypes, t.Margin)
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
	if len(t.model.Backbone) == 0 {
		return fmt.Errorf("Backboneが空です: 学習前に1層以上追加するべき")
	}

	if len(t.model.Prototypes) == 0 {
		return fmt.Errorf("Prototypesが未設定です: 学習前にSetClassPrototypes等で設定するべき")
	}

	if t.LR <= 0.0 {
		return fmt.Errorf("LRが不正(LR <= 0): LR = %v: LR > 0 であるべき", t.LR)
	}

	if len(t.workerRNGs) == 0 {
		return fmt.Errorf("workerRNGsが空です: NewTrainerのpは1以上であるべき")
	}

	if err := t.model.validateAscendingValues(); err != nil {
		return err
	}

	// gobロード後に SetSharedHyperparameters を忘れると学習時にnilパニックになるため、ここで弾く
	for i, layer := range t.model.Backbone {
		if d, ok := layer.(*Dense); ok && d.sharedHyperparameters == nil {
			return fmt.Errorf("layer %d: sharedHyperparameters が未設定です。学習前に Backbone.SetSharedHyperparameters を呼んでください", i)
		}
	}

	return nil
}

func SatisfiesUpdateCriterion(y *bitsx.Matrix, label int, prototypes bitsx.Matrices, margin float32) (bool, error) {
	if y == nil {
		return false, fmt.Errorf("yがnilです")
	}
	if len(prototypes) == 0 {
		return false, fmt.Errorf("prototypesが空です")
	}
	if label < 0 || label >= len(prototypes) {
		return false, fmt.Errorf("labelが範囲外: label = %d: 0 <= label < %d であるべき", label, len(prototypes))
	}

	t := prototypes[label]
	yMismatch, err := y.HammingDistance(t)
	if err != nil {
		return false, err
	}

	totalBits := y.Rows * y.Cols
	// margin は論文(BEP)の r と同じ ±1内積スケール。
	// ここでのロジットは一致ビット数(0..K)で、±1内積 = 2*一致数 - K だから、
	// 論文の r*K は一致数の差では r*K/2 に相当する。
	marginBits := int(float32(totalBits) * margin / 2)
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
