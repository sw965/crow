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

type Trainer struct {
	MiniBatchSize int
	LR     float32
	Margin float32

	model           Model
	rands           []*rand.Rand
	rand            *rand.Rand
	workerDelta     WorkerDelta
	aggregatedDelta SeqDelta
}

func NewTrainer(model Model, p int) *Trainer {
	rands := make([]*rand.Rand, p)
	workerDelta := make(WorkerDelta, p)
	backbone := model.Backbone
	numLayers := len(backbone)

	// ワーカーごとのバッファと乱数生成器の初期化
	for i := 0; i < p; i++ {
		sd := make(SeqDelta, numLayers)
		for l, layer := range backbone {
			sd[l] = layer.NewZerosDeltas()
		}
		workerDelta[i] = sd
		rands[i] = randx.NewPCGFromGlobalSeed()
	}

	// 集約用バッファの初期化
	aggregatedDelta := make(SeqDelta, numLayers)
	for l, layer := range backbone {
		aggregatedDelta[l] = layer.NewZerosDeltas()
	}

	return &Trainer{
		MiniBatchSize:   128,
		LR:              0.1,
		Margin:          0.0,
		model:           model,
		rands:           rands,
		rand:            randx.NewPCGFromGlobalSeed(),
		workerDelta:     workerDelta,
		aggregatedDelta: aggregatedDelta,
	}
}

func (t *Trainer) Train(xs bitsx.Matrices, labels []int) error {
	if err := t.Validate(); err != nil {
		return err
	}

	mbSize := t.MiniBatchSize
	if mbSize <= 0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	n := len(xs)
	if n < mbSize {
		mbSize = n
	}

	idxs := t.rand.Perm(n)
	for i := 0; i < n; i += mbSize {
		end := i + mbSize
		if end > n {
			end = n
		}

		miniIdxs := idxs[i:end]
		miniXs, err := slicesx.ElementsByIndices(xs, miniIdxs...)
		if err != nil {
			return err
		}

		miniLabels, err := slicesx.ElementsByIndices(labels, miniIdxs...)
		if err != nil {
			return err
		}

		seqDelta, err := t.ComputeSeqSignDelta(miniXs, miniLabels)
		if err != nil {
			return err
		}

		err = t.model.Backbone.Update(seqDelta, t.LR, t.rands)
		if err != nil {
			return err
		}
	}
	return nil
}

func (t *Trainer) ComputeSeqSignDelta(xs bitsx.Matrices, labels []int) (SeqDelta, error) {
	n := len(xs)
	if n > math.MaxInt16 {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	if n != len(labels) {
		return nil, fmt.Errorf("length mismatch: xs %d != labels %d", n, len(labels))
	}

	p := len(t.rands)
	t.workerDelta.Clear()
	backbone := t.model.Backbone
	prototypes := t.model.Prototypes

	err := parallel.For(n, p, func(workerId, idx int) error {
		rng := t.rands[workerId]
		x := xs[idx]
		label := labels[idx]

		y, backwards, err := backbone.Forwards(x, rng)
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
		_, err = backwards.Propagate(target, t.workerDelta[workerId])
		if err != nil {
			return err
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	err = t.workerDelta.Aggregate(t.aggregatedDelta)
	if err != nil {
		return nil, err
	}

	t.aggregatedDelta.Sign()
	return t.aggregatedDelta, nil
}

func (t *Trainer) Validate() error {
	if len(t.model.Backbone) == 0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	if len(t.model.Prototypes) == 0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	if t.LR <= 0.0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	if len(t.rands) == 0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	return nil
}

func SatisfiesUpdateCriterion(y *bitsx.Matrix, label int, prototypes bitsx.Matrices, margin float32) (bool, error) {
	t := prototypes[label]
	yMismatch, err := y.HammingDistance(t)
	if err != nil {
		return false, err
	}

	totalBits := y.Rows * y.Cols
	marginBits := int(float32(totalBits) * margin)
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
