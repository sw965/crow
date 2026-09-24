package binary

import (
	"testing"

	"github.com/sw965/omw/mathx/bitsx"
)

func TestMinPrototypeHammingDistance(t *testing.T) {
	newMatrix := func(setBits ...int) *bitsx.Matrix {
		m, err := bitsx.NewZerosMatrix(1, 8)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		for _, c := range setBits {
			if err := m.Set(0, c); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
		}
		return m
	}

	t.Run("正常_全ペアの最小値を返す", func(t *testing.T) {
		// 距離は a-b: 2, a-c: 5, b-c: 7
		a := newMatrix()
		b := newMatrix(0, 1)
		c := newMatrix(2, 3, 4, 5, 6)
		got, err := minPrototypeHammingDistance(bitsx.Matrices{a, b, c})
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if got != 2 {
			t.Errorf("値の不一致: got = %d, want = 2", got)
		}
	})

	t.Run("正常_温度計の最小距離は隣接段の距離", func(t *testing.T) {
		protos, err := bitsx.NewThermometerMatrices(11, 1, 1024)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		got, err := minPrototypeHammingDistance(protos)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		// 1024 / (11-1) = 102.4 の切り捨て
		if got != 102 {
			t.Errorf("値の不一致: got = %d, want = 102", got)
		}
	})

	for _, test := range []struct {
		name   string
		protos bitsx.Matrices
	}{
		{name: "境界_空"},
		{name: "境界_1個", protos: bitsx.Matrices{newMatrix(0)}},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := minPrototypeHammingDistance(test.protos)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if got != 0 {
				t.Errorf("値の不一致: got = %d, want = 0", got)
			}
		})
	}

	t.Run("異常_形状が不一致", func(t *testing.T) {
		other, err := bitsx.NewZerosMatrix(1, 16)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if _, err := minPrototypeHammingDistance(bitsx.Matrices{newMatrix(), other}); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}
