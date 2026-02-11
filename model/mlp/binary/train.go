package binary

import (
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"math/rand/v2"
	"fmt"
	"github.com/sw965/omw/slicesx"
)

type Trainer struct {
	model        Sequence
	Prototypes   bitsx.Matrices
	LR           float32

	computer *SeqSignDeltaComputer
	rands    []*rand.Rand
	rand     *rand.Rand
}

func NewTrainer(model Sequence, margin float32, p int) *Trainer {
	rands := make([]*rand.Rand, p)
	for i := range p {
		rands[i] = randx.NewPCGFromGlobalSeed()
	}
	computer := NewSeqSignDeltaComputer(model, margin, p)
	return &Trainer{
		model:    model,
		computer: computer,
		rands:    rands,
		rand:     randx.NewPCGFromGlobalSeed(),
	}
}

func (t *Trainer) Fit(xs bitsx.Matrices, labels []int, miniBatchSize int) error {
	if err := t.Validate(); err != nil {
		return err
	}

	n := len(xs)
	if miniBatchSize <= 0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	if n < miniBatchSize {
		miniBatchSize = n
	}

	idxs := t.rand.Perm(n)
	for i := 0; i < n; i += miniBatchSize {
		end := i + miniBatchSize
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

		seqDelta, err := t.computer.Compute(t.model, miniXs, miniLabels, t.Prototypes)
		if err != nil {
			return err
		}

		err = t.model.Update(seqDelta, t.LR, t.rands)
		if err != nil {
			return err
		}
	}
	return nil
}

func (t *Trainer) Validate() error {
	if len(t.model) == 0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	if len(t.Prototypes) == 0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	if t.LR <= 0.0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	if t.computer == nil {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	if len(t.rands) == 0 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	if t.rands == nil {
		return fmt.Errorf("後でエラーメッセージを書く")
	}
	return nil
}