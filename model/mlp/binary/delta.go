package binary

import (
	"cmp"
	"fmt"
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