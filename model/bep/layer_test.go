package bep_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/sw965/crow/model/bep"
	"github.com/sw965/omw/mathx/bitsx"
)

func TestSequenceSetGroupSize(t *testing.T) {
	t.Run("正常_全層に設定", func(t *testing.T) {
		model, rng := newTestModel(t)
		if err := model.AppendDenseLayer(16, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if err := model.Backbone.SetGroupSize(2); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		for i, layer := range model.Backbone {
			if d := layer.(*bep.Dense); d.GroupSize != 2 {
				t.Errorf("layer %d: GroupSize の不一致: got = %d, want = 2", i, d.GroupSize)
			}
		}
	})

	t.Run("異常_0以下なら値を変えない", func(t *testing.T) {
		model, _ := newTestModel(t)
		d := model.Backbone[0].(*bep.Dense)
		before := d.GroupSize
		if err := model.Backbone.SetGroupSize(0); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if d.GroupSize != before {
			t.Errorf("エラーなのに GroupSize が変わった: got = %d, want = %d", d.GroupSize, before)
		}
	})
}

// assertDenseConsistent は、W = sign(H) かつ WT = W の転置であることを確認する。
func assertDenseConsistent(t *testing.T, d *bep.Dense) {
	t.Helper()
	cols := d.W.Cols()
	for r := range d.W.Rows() {
		for c := range cols {
			bit, err := d.W.Bit(r, c)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			want := uint64(0)
			if d.H[r*cols+c] >= 0 {
				want = 1
			}
			if bit != want {
				t.Fatalf("W が sign(H) と不一致 (r = %d, c = %d): got = %d, want = %d", r, c, bit, want)
			}
		}
	}
	wt, err := d.W.Transpose()
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if !wt.Equal(d.WT) {
		t.Fatal("WT が W の転置と不一致")
	}
}

func TestDenseUpdate(t *testing.T) {
	clip := func(v int) int8 { return int8(max(math.MinInt8, min(v, math.MaxInt8))) }

	for _, shape := range [][2]int{{8, 64}, {16, 100}, {33, 37}} {
		t.Run(fmt.Sprintf("正常_lrHalfPowが0なら全要素を更新_%dx%d", shape[0], shape[1]), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(1, 2))
			d, err := bep.NewDense(shape[0], shape[1], rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			deltas := d.NewZerosDeltas()
			for i := range deltas[0] {
				// クリップと符号反転の両方を通すため、H の値域を超える大きさも混ぜる
				deltas[0][i] = int16(rng.IntN(401) - 200)
			}
			h0 := slices.Clone(d.H)

			if err := d.Update(deltas, 0, rng); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			for i, v := range d.H {
				if want := clip(int(h0[i]) + int(deltas[0][i])); v != want {
					t.Fatalf("H の不一致 (i = %d): got = %d, want = %d", i, v, want)
				}
			}
			assertDenseConsistent(t, d)
		})
	}

	for _, tt := range []struct {
		lrHalfPow int
		wantP     float64
	}{
		{lrHalfPow: 1, wantP: 0.5},
		{lrHalfPow: 3, wantP: 0.125},
	} {
		t.Run(fmt.Sprintf("正常_更新確率_lrHalfPow=%d", tt.lrHalfPow), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(3, 4))
			// 列数を64の倍数にしないことで、端数ワードの範囲外ビットを更新しないことも確認する
			d, err := bep.NewDense(64, 1000, rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			deltas := d.NewZerosDeltas()
			for i := range deltas[0] {
				// 0 を除き、更新されたかどうかを H の変化で判別できるようにする
				if rng.IntN(2) == 0 {
					deltas[0][i] = 1
				} else {
					deltas[0][i] = -1
				}
			}
			h0 := slices.Clone(d.H)

			if err := d.Update(deltas, tt.lrHalfPow, rng); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			updated := 0
			for i, v := range d.H {
				switch v {
				case h0[i]:
				case clip(int(h0[i]) + int(deltas[0][i])):
					updated++
				default:
					t.Fatalf("H が更新前とも更新後とも一致しない (i = %d): got = %d, h0 = %d, delta = %d", i, v, h0[i], deltas[0][i])
				}
			}
			// N=64000。tol=0.02 は p=0.5 で約10σ、p=0.125 で約15σ
			if gotP := float64(updated) / float64(len(d.H)); math.Abs(gotP-tt.wantP) > 0.02 {
				t.Errorf("更新確率の不一致: got = %f, want = %f", gotP, tt.wantP)
			}
			assertDenseConsistent(t, d)
		})
	}

	t.Run("異常_lrHalfPowが負", func(t *testing.T) {
		rng := rand.New(rand.NewPCG(5, 6))
		d, err := bep.NewDense(4, 10, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if err := d.Update(d.NewZerosDeltas(), -1, rng); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestDenseForwardNoise(t *testing.T) {
	const (
		rows  = 4
		xCols = 100
		wRows = 300
	)
	newDense := func(t *testing.T, maxAbsNoise int) (*bep.Dense, *bitsx.Matrix) {
		t.Helper()
		rng := rand.New(rand.NewPCG(11, 12))
		d, err := bep.NewDense(wRows, xCols, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		d.MaxAbsNoise = maxAbsNoise
		x, err := bitsx.NewRandMatrix(rows, xCols, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		return d, x
	}

	t.Run("正常_MaxAbsNoiseが0ならPredictと一致し乱数を消費しない", func(t *testing.T) {
		d, x := newDense(t, 0)
		rng := rand.New(rand.NewPCG(1, 2))
		y, _, err := d.Forward(x, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		want, err := d.Predict(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !y.Equal(want) {
			t.Error("MaxAbsNoiseが0の Forward が Predict と一致しない")
		}
		// biaslab の差分テストは、ノイズ無効時に乱数を消費しないことを前提にしている
		if rng.Uint64() != rand.New(rand.NewPCG(1, 2)).Uint64() {
			t.Error("MaxAbsNoiseが0なのに乱数を消費した")
		}
	})

	t.Run("正常_MaxAbsNoiseを超える位置のニューロンは反転しない", func(t *testing.T) {
		const maxAbsNoise = 10
		d, x := newDense(t, maxAbsNoise)
		u, err := x.Dot(d.W)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		rng := rand.New(rand.NewPCG(3, 4))
		flipped := 0
		for range 20 {
			y, _, err := d.Forward(x, rng)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			for r := range rows {
				for c := range wRows {
					z := 2*u[r*wRows+c] - xCols
					bit, err := y.Bit(r, c)
					if err != nil {
						t.Fatalf("予期せぬエラー: %v", err)
					}
					noiseless := uint64(0)
					if z >= 0 {
						noiseless = 1
					}
					if bit != noiseless {
						if z > maxAbsNoise || z < -maxAbsNoise {
							t.Fatalf("|z| = %d が MaxAbsNoise %d を超えているのに反転した (r = %d, c = %d)", z, maxAbsNoise, r, c)
						}
						flipped++
					}
				}
			}
		}
		if flipped == 0 {
			t.Error("ノイズで反転したニューロンが1つも無い")
		}
	})
}

func TestSetNoiseScale(t *testing.T) {
	newSeq := func(t *testing.T) (bep.Sequence, *bep.Dense, *bep.Dense) {
		t.Helper()
		rng := rand.New(rand.NewPCG(21, 22))
		first, err := bep.NewDense(64, 784, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		second, err := bep.NewDense(10, 64, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		return bep.Sequence{first, second}, first, second
	}

	t.Run("正常_Denseの倍率3/4", func(t *testing.T) {
		_, first, _ := newSeq(t)
		if err := first.SetNoiseScale(3, 4); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if first.MaxAbsNoise != 36 {
			t.Errorf("MaxAbsNoise の不一致: got = %d, want = 36", first.MaxAbsNoise)
		}
	})

	t.Run("異常_Denseの倍率が不正なら値を変えない", func(t *testing.T) {
		_, first, _ := newSeq(t)
		before := first.MaxAbsNoise
		if err := first.SetNoiseScale(1, 0); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if first.MaxAbsNoise != before {
			t.Errorf("エラーなのに MaxAbsNoise が変わった: got = %d, want = %d", first.MaxAbsNoise, before)
		}
	})

	t.Run("正常_Sequenceで全層に設定", func(t *testing.T) {
		seq, first, second := newSeq(t)
		if err := seq.SetNoiseScale(1, 1); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		// isqrt(784) = 28、isqrt(64) = 8 に √3 ≈ 1.732 を掛けて切り捨て
		if first.MaxAbsNoise != 48 || second.MaxAbsNoise != 13 {
			t.Errorf("MaxAbsNoise の不一致: got = (%d, %d), want = (48, 13)", first.MaxAbsNoise, second.MaxAbsNoise)
		}
	})

	t.Run("異常_Sequenceで一部の層が範囲外なら全層を変えない", func(t *testing.T) {
		seq, first, second := newSeq(t)
		before1, before2 := first.MaxAbsNoise, second.MaxAbsNoise
		// 784入力の層は 484 で収まるが、64入力の層は 138 で入力数を超える
		err := seq.SetNoiseScale(10, 1)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "layer 1") {
			t.Errorf("エラーメッセージに層番号が無い: %s", err.Error())
		}
		if first.MaxAbsNoise != before1 || second.MaxAbsNoise != before2 {
			t.Errorf("エラーなのに値が変わった: got = (%d, %d), want = (%d, %d)",
				first.MaxAbsNoise, second.MaxAbsNoise, before1, before2)
		}
	})
}
