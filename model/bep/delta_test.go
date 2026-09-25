package bep_test

import (
	"slices"
	"testing"

	"github.com/sw965/crow/model/bep"
)

func TestDelta(t *testing.T) {
	t.Run("正常_Sign", func(t *testing.T) {
		d := bep.Delta{5, -3, 0, 100, -1}
		d.Sign()
		want := bep.Delta{1, -1, 0, 1, -1}
		if !slices.Equal(d, want) {
			t.Errorf("値の不一致: got = %v, want = %v", d, want)
		}
	})
}
