package binary

import (
	"cmp"
	"fmt"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	"math"
	"math/rand/v2"
	"slices"
)

type Delta []int16

func (d Delta) Add(other Delta) error {
	if len(d) != len(other) {
		return fmt.Errorf("後でエラーメッセージを書く")
	}
	for i, v := range other {
		d[i] += v
	}
	return nil
}

func (d Delta) Sign() {
	for i, v := range d {
		d[i] = int16(cmp.Compare(v, 0))
	}
}

type Deltas []Delta

func (ds Deltas) ZerosLike() Deltas {
	zeros := make(Deltas, len(ds))
	for i, d := range ds {
		zeros[i] = make(Delta, len(d))
	}
	return zeros
}

func (ds Deltas) Add(other Deltas) error {
	if len(ds) != len(other) {
		return fmt.Errorf("後でエラーメッセージを書く")
	}
	for i, d := range other {
		err := ds[i].Add(d)
		if err != nil {
			return err
		}
	}
	return nil
}

func (ds Deltas) Sign() {
	for _, d := range ds {
		d.Sign()
	}
}

func (ds Deltas) Clear() {
	for i := range ds {
		clear(ds[i])
	}
}

func (ds Deltas) Clone() Deltas {
	c := make(Deltas, len(ds))
	for i := range ds {
		c[i] = slices.Clone(ds[i])
	}
	return c
}

type SeqDelta []Deltas

func (sd SeqDelta) Add(other SeqDelta) error {
	if len(sd) != len(other) {
		return fmt.Errorf("sequence delta length mismatch: %d != %d", len(sd), len(other))
	}
	for i := range sd {
		if err := sd[i].Add(other[i]); err != nil {
			return err
		}
	}
	return nil
}

func (sd SeqDelta) Sign() {
	for i := range sd {
		sd[i].Sign()
	}
}

func (sd SeqDelta) Clone() SeqDelta {
	c := make(SeqDelta, len(sd))
	for i := range c {
		c[i] = sd[i].Clone()
	}
	return c
}

func (sd SeqDelta) Clear() {
	for _, d := range sd {
		d.Clear()
	}
}

type WorkerDelta []SeqDelta

func (wd WorkerDelta) Clear() {
	for i := range wd {
		wd[i].Clear()
	}
}

func (wd WorkerDelta) Aggregate(dst SeqDelta) error {
	if len(wd) == 0 {
		return fmt.Errorf("worker delta is empty")
	}

	dst.Clear()
	for _, sd := range wd {
		if err := dst.Add(sd); err != nil {
			return err
		}
	}
	return nil
}

type SeqSignDeltaComputer struct {
	margin          float32
	rands           []*rand.Rand
	workerDelta     WorkerDelta
	aggregatedDelta SeqDelta
}

func NewSeqSignDeltaComputer(s Sequence, margin float32, p int) *SeqSignDeltaComputer {
	rands := make([]*rand.Rand, p)
	workerDelta := make(WorkerDelta, p)

	for i := 0; i < p; i++ {
		rands[i] = randx.NewPCGFromGlobalSeed()
		sd := make(SeqDelta, len(s))
		for l, layer := range s {
			sd[l] = layer.NewZerosDeltas()
		}
		workerDelta[i] = sd
	}

	aggregatedDelta := make(SeqDelta, len(s))
	for l, layer := range s {
		aggregatedDelta[l] = layer.NewZerosDeltas()
	}

	return &SeqSignDeltaComputer{
		margin:      margin,
		rands:       rands,
		workerDelta: workerDelta,
		aggregatedDelta:aggregatedDelta,
	}
}

func (s *SeqSignDeltaComputer) Compute(seq Sequence, xs bitsx.Matrices, labels []int, prototypes bitsx.Matrices) (SeqDelta, error) {
	n := len(xs)
	if n > math.MaxInt16 {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	if n != len(labels) {
		return nil, fmt.Errorf("length mismatch: xs %d != labels %d", n, len(labels))
	}

	s.workerDelta.Clear()
	p := len(s.rands)

	// workerIdをworkerIに変える
	err := parallel.For(n, p, func(workerId, idx int) error {
		rng := s.rands[workerId]
		x := xs[idx]
		label := labels[idx]

		y, backwards, err := seq.Forwards(x, rng)
		if err != nil {
			return err
		}

		shouldUpdate, err := SatisfiesUpdateCriterion(y, label, prototypes, s.margin)
		if err != nil {
			return err
		}

		if !shouldUpdate {
			return nil
		}

		t := prototypes[label]
		_, err = backwards.Propagate(t, s.workerDelta[workerId])
		if err != nil {
			return err
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	err = s.workerDelta.Aggregate(s.aggregatedDelta)
	if err != nil {
		return nil, err
	}

	s.aggregatedDelta.Sign()
	return s.aggregatedDelta, nil
}

func SatisfiesUpdateCriterion(y bitsx.Matrix, label int, prototypes bitsx.Matrices, margin float32) (bool, error) {
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
