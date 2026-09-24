package bep_test

import (
	"slices"
	"testing"

	"github.com/sw965/crow/model/bep"
)

func TestDelta(t *testing.T) {
	t.Run("正常_Add", func(t *testing.T) {
		d := bep.Delta{1, -2, 3}
		if err := d.Add(bep.Delta{10, 20, 30}); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		want := bep.Delta{11, 18, 33}
		if !slices.Equal(d, want) {
			t.Errorf("値の不一致: got = %v, want = %v", d, want)
		}
	})

	t.Run("異常_Add_長さ不一致", func(t *testing.T) {
		d := bep.Delta{1, 2}
		err := d.Add(bep.Delta{1})
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})

	t.Run("正常_Sign", func(t *testing.T) {
		d := bep.Delta{5, -3, 0, 100, -1}
		d.Sign()
		want := bep.Delta{1, -1, 0, 1, -1}
		if !slices.Equal(d, want) {
			t.Errorf("値の不一致: got = %v, want = %v", d, want)
		}
	})

	t.Run("正常_Aggregate", func(t *testing.T) {
		sd1 := bep.SeqDelta{bep.Deltas{bep.Delta{1, 2}}}
		sd2 := bep.SeqDelta{bep.Deltas{bep.Delta{10, 20}}}
		dst := bep.SeqDelta{bep.Deltas{bep.Delta{99, 99}}} // Aggregate前にクリアされる事も確認

		sds := bep.SeqDeltas{sd1, sd2}
		if err := sds.Aggregate(dst); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		want := bep.Delta{11, 22}
		if !slices.Equal(dst[0][0], want) {
			t.Errorf("値の不一致: got = %v, want = %v", dst[0][0], want)
		}
	})

	t.Run("異常_Aggregate_空", func(t *testing.T) {
		sds := bep.SeqDeltas{}
		err := sds.Aggregate(bep.SeqDelta{})
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}
