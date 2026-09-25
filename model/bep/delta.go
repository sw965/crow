package bep

import (
	"cmp"
)

type Delta []int16

func (d Delta) Sign() {
	for i, v := range d {
		d[i] = int16(cmp.Compare(v, 0))
	}
}

type Deltas []Delta

func (ds Deltas) Sign() {
	for _, d := range ds {
		d.Sign()
	}
}

type SeqDelta []Deltas

func (sd SeqDelta) Sign() {
	for i := range sd {
		sd[i].Sign()
	}
}
