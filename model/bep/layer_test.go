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

func TestSequenceSetMaxUpdateAbsZScale(t *testing.T) {
	t.Run("正常_全層に入力数に応じた値を設定", func(t *testing.T) {
		model, rng := newTestModel(t)
		if err := model.AppendDenseLayer(16, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if err := model.Backbone.SetMaxUpdateAbsZScale(1, 2); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		// 入力数 64 → isqrt 8 × 1/2 = 4、入力数 32 → isqrt 5 × 1/2 = 2
		for i, want := range []int{4, 2} {
			if d := model.Backbone[i].(*bep.Dense); d.MaxUpdateAbsZ != want {
				t.Errorf("layer %d: MaxUpdateAbsZ の不一致: got = %d, want = %d", i, d.MaxUpdateAbsZ, want)
			}
		}
	})

	t.Run("異常_倍率が不正なら値を変えない", func(t *testing.T) {
		for _, scale := range [][2]int{{1, 0}, {-1, 1}, {100, 1}} {
			model, _ := newTestModel(t)
			d := model.Backbone[0].(*bep.Dense)
			before := d.MaxUpdateAbsZ
			if err := model.Backbone.SetMaxUpdateAbsZScale(scale[0], scale[1]); err == nil {
				t.Fatalf("倍率 %d/%d: エラーを期待したが、nilが返された", scale[0], scale[1])
			}
			if d.MaxUpdateAbsZ != before {
				t.Errorf("倍率 %d/%d: エラーなのに MaxUpdateAbsZ が変わった: got = %d, want = %d", scale[0], scale[1], d.MaxUpdateAbsZ, before)
			}
		}
	})
}

func TestDenseUpdateSelection(t *testing.T) {
	const xRows, fanIn, outs = 2, 70, 90
	rng := rand.New(rand.NewPCG(91, 92))
	d, err := bep.NewDense(outs, fanIn, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	// 入力数 70 → isqrt 8 × 1/4 = 2
	if d.MaxUpdateAbsZ != 2 || d.MarginAbsZ != 2 {
		t.Errorf("既定値の不一致: got = (MaxUpdateAbsZ %d, MarginAbsZ %d), want = (2, 2)", d.MaxUpdateAbsZ, d.MarginAbsZ)
	}
	setRandomBias(d, rng)
	d.MaxUpdateAbsZ = 8
	d.MarginAbsZ = 6

	x, err := bitsx.NewRandMatrix(xRows, fanIn, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	tgt, err := bitsx.NewRandMatrix(xRows, outs, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	_, bw, err := d.Forward(x)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	// 行ごとに記録し、どのニューロンが直されたかをバイアスのデルタから読む
	selected, skipped, pushed, kept := 0, 0, 0, 0
	for r := range xRows {
		xr, err := bitsx.NewZerosMatrix(1, fanIn)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		tr, err := bitsx.NewZerosMatrix(1, outs)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		copyBitsRow(t, xr, x, r)
		copyBitsRow(t, tr, tgt, r)
		_, bwr, err := d.Forward(xr)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		rec, err := d.NewBatchRecord(1, 1)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if _, err := bwr(tr, rec, 0); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		deltas, err := d.BatchDeltas(rec)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		for j := range outs {
			z := naiveZ(t, d, xr, 0, j)
			tBit := mustBit(t, tr, 0, j)
			// 食い違いは |z| <= MaxUpdateAbsZ、正解は |z| < MarginAbsZ のとき、どちらも目標の向きへ押す
			var want int16
			switch mismatch := signBitOf(z) != tBit; {
			case mismatch && absOf(z) <= d.MaxUpdateAbsZ:
				want = int16(2*int(tBit) - 1)
				selected++
			case mismatch:
				skipped++
			case absOf(z) < d.MarginAbsZ:
				want = int16(2*int(tBit) - 1)
				pushed++
			default:
				kept++
			}
			if got := deltas[1][j]; got != want {
				t.Fatalf("r = %d, j = %d (z = %d, t = %d): バイアスのデルタの不一致: got = %d, want = %d", r, j, z, tBit, got, want)
			}
		}
	}
	if selected == 0 || skipped == 0 || pushed == 0 || kept == 0 {
		t.Fatalf("しきい値の両側にニューロンがなく、テストになっていない: selected = %d, skipped = %d, pushed = %d, kept = %d",
			selected, skipped, pushed, kept)
	}
	if _, err := bw(tgt, otherRecord{}, 0); err == nil {
		t.Error("backward がエラーにならない")
	}
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

func newZeroDeltas(d *bep.Dense) bep.Deltas {
	return bep.Deltas{make(bep.Delta, d.W.Rows()*d.W.Cols()), make(bep.Delta, d.W.Rows())}
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
			deltas := newZeroDeltas(d)
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
			deltas := newZeroDeltas(d)
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
		if err := d.Update(newZeroDeltas(d), -1, rng); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestSmallFanInDefaults(t *testing.T) {
	t.Run("正常_Denseは入力数9でも既定値が1以上で学習を始められる", func(t *testing.T) {
		model := bep.Model{XRows: 1, XCols: 9}
		rng := rand.New(rand.NewPCG(1, 2))
		if err := model.AppendDenseLayer(8, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if err := model.SetClassPrototypes(2, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		d := model.Backbone[0].(*bep.Dense)
		// 入力数 9 → isqrt 3 × 1/4 は切り捨てで 0 になるが、最低 1
		if d.MaxUpdateAbsZ != 1 || d.MarginAbsZ != 1 {
			t.Errorf("既定値の不一致: got = (MaxUpdateAbsZ %d, MarginAbsZ %d), want = (1, 1)", d.MaxUpdateAbsZ, d.MarginAbsZ)
		}
		if err := newTestTrainer(t, &model).Validate(); err != nil {
			t.Errorf("予期せぬエラー: %v", err)
		}
	})

	t.Run("正常_ProductDenseは入力数20でもマージンが1以上", func(t *testing.T) {
		p, err := bep.NewProductDense(10, 20, 2, rand.New(rand.NewPCG(1, 2)))
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		// 入力数 20 → isqrt 4 × 1/8 は切り捨てで 0 になるが、最低 1
		if p.MarginAbsZ != 1 {
			t.Errorf("MarginAbsZ の不一致: got = %d, want = 1", p.MarginAbsZ)
		}
	})
}

func TestDenseForwardMatchesPredict(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	d, err := bep.NewDense(300, 100, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	setRandomBias(d, rng)
	x, err := bitsx.NewRandMatrix(4, 100, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	y, _, err := d.Forward(x)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	want, err := d.Predict(x)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if !y.Equal(want) {
		t.Error("Forward の出力が Predict と一致しない")
	}
}

func TestSetMarginAbsZScale(t *testing.T) {
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

	t.Run("正常_Denseの倍率1/2", func(t *testing.T) {
		_, first, _ := newSeq(t)
		if err := first.SetMarginAbsZScale(1, 2); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		// isqrt(784) = 28 × 1/2
		if first.MarginAbsZ != 14 {
			t.Errorf("MarginAbsZ の不一致: got = %d, want = 14", first.MarginAbsZ)
		}
	})

	t.Run("正常_倍率0でマージンを無効にできる", func(t *testing.T) {
		_, first, _ := newSeq(t)
		if err := first.SetMarginAbsZScale(0, 1); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if first.MarginAbsZ != 0 {
			t.Errorf("MarginAbsZ の不一致: got = %d, want = 0", first.MarginAbsZ)
		}
		if err := first.Validate(); err != nil {
			t.Errorf("予期せぬエラー: %v", err)
		}
	})

	t.Run("異常_Denseの倍率が不正なら値を変えない", func(t *testing.T) {
		_, first, _ := newSeq(t)
		before := first.MarginAbsZ
		if err := first.SetMarginAbsZScale(1, 0); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if first.MarginAbsZ != before {
			t.Errorf("エラーなのに MarginAbsZ が変わった: got = %d, want = %d", first.MarginAbsZ, before)
		}
	})

	t.Run("正常_Sequenceで全層に設定", func(t *testing.T) {
		seq, first, second := newSeq(t)
		if err := seq.SetMarginAbsZScale(1, 1); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if first.MarginAbsZ != 28 || second.MarginAbsZ != 8 {
			t.Errorf("MarginAbsZ の不一致: got = (%d, %d), want = (28, 8)", first.MarginAbsZ, second.MarginAbsZ)
		}
	})

	t.Run("異常_Sequenceで一部の層が範囲外なら全層を変えない", func(t *testing.T) {
		seq, first, second := newSeq(t)
		before1, before2 := first.MarginAbsZ, second.MarginAbsZ
		// 784入力の層は 1120 で |z| の最大値 1568 に収まるが、64入力の層は 320 で最大値 128 を超える
		err := seq.SetMarginAbsZScale(40, 1)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "layer 1") || !strings.Contains(err.Error(), "MarginAbsZ") {
			t.Errorf("エラーメッセージに層番号か MarginAbsZ が無い: %s", err.Error())
		}
		if first.MarginAbsZ != before1 || second.MarginAbsZ != before2 {
			t.Errorf("エラーなのに値が変わった: got = (%d, %d), want = (%d, %d)",
				first.MarginAbsZ, second.MarginAbsZ, before1, before2)
		}
	})
}

func TestDenseBias(t *testing.T) {
	newDenseAndInput := func(t *testing.T) (*bep.Dense, *bitsx.Matrix, *rand.Rand) {
		t.Helper()
		rng := rand.New(rand.NewPCG(31, 32))
		d, err := bep.NewDense(40, 37, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		x, err := bitsx.NewRandMatrix(3, 37, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		return d, x, rng
	}

	t.Run("正常_初期値は0", func(t *testing.T) {
		d, _, _ := newDenseAndInput(t)
		if len(d.Bias) != d.W.Rows() {
			t.Fatalf("len(Bias) の不一致: got = %d, want = %d", len(d.Bias), d.W.Rows())
		}
		for j, b := range d.Bias {
			if b != 0 {
				t.Fatalf("Bias[%d] = %d: 初期値は0であるべき", j, b)
			}
		}
	})

	t.Run("正常_Predictはバイアス込みの符号", func(t *testing.T) {
		d, x, rng := newDenseAndInput(t)
		for j := range d.Bias {
			d.Bias[j] = int32(rng.IntN(2*37+1) - 37)
		}
		u, err := x.Dot(d.W)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		y, err := d.Predict(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		rows := d.W.Rows()
		for r := range x.Rows() {
			for j := range rows {
				z := 2*u[r*rows+j] - d.W.Cols() + int(d.Bias[j])
				want := uint64(0)
				if z >= 0 {
					want = 1
				}
				got, err := y.Bit(r, j)
				if err != nil {
					t.Fatalf("予期せぬエラー: %v", err)
				}
				if got != want {
					t.Fatalf("出力の不一致 (r = %d, j = %d, z = %d): got = %d, want = %d", r, j, z, got, want)
				}
			}
		}
	})

	t.Run("正常_バイアスのデルタは更新対象の希望出力", func(t *testing.T) {
		rng := rand.New(rand.NewPCG(41, 42))
		d, err := bep.NewDense(200, 37, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		// このテストは食い違ったニューロンだけを直すことを見るので、マージンは無効にする
		d.MarginAbsZ = 0
		// 入力を1行にして、ニューロンごとの寄与が行をまたいで打ち消し合わないようにする
		x, err := bitsx.NewRandMatrix(1, 37, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		y, bw, err := d.Forward(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		tgt, err := bitsx.NewRandMatrix(1, y.Cols(), rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		rec, err := d.NewBatchRecord(1, 1)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if _, err := bw(tgt, rec, 0); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		deltas, err := d.BatchDeltas(rec)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if len(deltas[1]) != d.W.Rows() {
			t.Fatalf("バイアスのデルタ長の不一致: got = %d, want = %d", len(deltas[1]), d.W.Rows())
		}

		cols := d.W.Cols()
		selected := 0
		for j := range d.W.Rows() {
			yBit, err := y.Bit(0, j)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			tBit, err := tgt.Bit(0, j)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			// 更新対象に選ばれたニューロンだけ、重みのデルタ行が非ゼロになる
			isSelected := slices.ContainsFunc(deltas[0][j*cols:(j+1)*cols], func(v int16) bool { return v != 0 })
			if !isSelected {
				if deltas[1][j] != 0 {
					t.Fatalf("j = %d: 更新対象でないのにバイアスのデルタが %d", j, deltas[1][j])
				}
				continue
			}
			selected++
			if yBit == tBit {
				t.Fatalf("j = %d: 出力が目標と一致しているのに更新対象になった", j)
			}
			want := int16(-1)
			if tBit == 1 {
				want = 1
			}
			if deltas[1][j] != want {
				t.Fatalf("j = %d: バイアスのデルタの不一致: got = %d, want = %d", j, deltas[1][j], want)
			}
		}
		if selected == 0 {
			t.Fatal("更新対象が1つも無く、テストが何も確かめていない")
		}
	})

	t.Run("正常_Updateはデルタの符号で動き入力数で止まる", func(t *testing.T) {
		d, _, rng := newDenseAndInput(t)
		fanIn := int32(d.W.Cols())
		d.Bias[0] = fanIn
		d.Bias[1] = -fanIn
		d.Bias[2] = 5
		deltas := newZeroDeltas(d)
		deltas[1][0] = 1
		deltas[1][1] = -1
		deltas[1][2] = -1
		deltas[1][3] = 1
		if err := d.Update(deltas, 0, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		want := []int32{fanIn, -fanIn, 4, 1}
		for j, w := range want {
			if d.Bias[j] != w {
				t.Errorf("Bias[%d] の不一致: got = %d, want = %d", j, d.Bias[j], w)
			}
		}
		for j := 4; j < len(d.Bias); j++ {
			if d.Bias[j] != 0 {
				t.Fatalf("デルタ0なのに Bias[%d] が動いた: %d", j, d.Bias[j])
			}
		}
	})
}

func mustBit(t *testing.T, m *bitsx.Matrix, r, c int) uint64 {
	t.Helper()
	b, err := m.Bit(r, c)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	return b
}

// src の r 行目を、1行の dst へ写す
func copyBitsRow(t *testing.T, dst, src *bitsx.Matrix, r int) {
	t.Helper()
	for c := range src.Cols() {
		if mustBit(t, src, r, c) == 1 {
			if err := dst.Set(0, c); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
		}
	}
}

type otherRecord struct{}

func (otherRecord) Clear() error { return nil }

func TestDenseBatchDeltas(t *testing.T) {
	const xRows, fanIn, outs = 2, 70, 90
	newDense := func(t *testing.T) (*bep.Dense, *rand.Rand) {
		t.Helper()
		rng := rand.New(rand.NewPCG(51, 52))
		d, err := bep.NewDense(outs, fanIn, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		return d, rng
	}
	// 1サンプル分の Forward と、目標をランダムに作る
	newSample := func(t *testing.T, d *bep.Dense, rng *rand.Rand) (*bitsx.Matrix, *bitsx.Matrix, bep.Backward) {
		t.Helper()
		x, err := bitsx.NewRandMatrix(xRows, fanIn, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		_, bw, err := d.Forward(x)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		tgt, err := bitsx.NewRandMatrix(xRows, outs, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		return x, tgt, bw
	}
	deltasOf := func(t *testing.T, d *bep.Dense, rec bep.BatchRecord) bep.Deltas {
		t.Helper()
		ds, err := d.BatchDeltas(rec)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		return bep.Deltas{slices.Clone(ds[0]), slices.Clone(ds[1])}
	}

	t.Run("正常_複数サンプルの記録は1サンプルずつの合計と一致", func(t *testing.T) {
		d, rng := newDense(t)
		const n = 7
		batch, err := d.NewBatchRecord(n, xRows)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		want := newZeroDeltas(d)
		for s := range n {
			_, tgt, bw := newSample(t, d, rng)
			single, err := d.NewBatchRecord(1, xRows)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if _, err := bw(tgt, single, 0); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			for k, dl := range deltasOf(t, d, single) {
				for i, v := range dl {
					want[k][i] += v
				}
			}
			if _, err := bw(tgt, batch, s); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
		}
		got := deltasOf(t, d, batch)
		for k := range got {
			if !slices.Equal(got[k], want[k]) {
				t.Fatalf("Deltas[%d] が1サンプルずつの合計と一致しない", k)
			}
		}
	})

	t.Run("正常_1サンプルの重みデルタは入力と希望出力の一致で±1", func(t *testing.T) {
		d, rng := newDense(t)
		x, tgt, bw := newSample(t, d, rng)
		rec, err := d.NewBatchRecord(1, xRows)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if _, err := bw(tgt, rec, 0); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		got := deltasOf(t, d, rec)

		// 各行で選ばれたニューロンを、行ごとの記録から復元する代わりに、1行ずつの入力で求め直して素朴に合計する
		want := newZeroDeltas(d)
		for r := range xRows {
			xr, err := bitsx.NewZerosMatrix(1, fanIn)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			tr, err := bitsx.NewZerosMatrix(1, outs)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			copyBitsRow(t, xr, x, r)
			copyBitsRow(t, tr, tgt, r)
			_, bwr, err := d.Forward(xr)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			recr, err := d.NewBatchRecord(1, 1)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if _, err := bwr(tr, recr, 0); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			selected := deltasOf(t, d, recr)[1] // 1行なので、選ばれたニューロンは ±1、それ以外は 0
			for j := range outs {
				if selected[j] == 0 {
					continue
				}
				tBit := mustBit(t, tr, 0, j)
				want[1][j] += selected[j]
				for c := range fanIn {
					xBit := mustBit(t, xr, 0, c)
					v := int16(-1)
					if xBit == tBit {
						v = 1
					}
					want[0][j*fanIn+c] += v
				}
			}
		}
		for k := range got {
			if !slices.Equal(got[k], want[k]) {
				t.Fatalf("Deltas[%d] が素朴な計算と一致しない", k)
			}
		}
	})

	t.Run("正常_Clearで前のバッチの記録が消える", func(t *testing.T) {
		d, rng := newDense(t)
		rec, err := d.NewBatchRecord(1, xRows)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		_, tgt, bw := newSample(t, d, rng)
		if _, err := bw(tgt, rec, 0); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if err := rec.Clear(); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		for k, dl := range deltasOf(t, d, rec) {
			if slices.ContainsFunc(dl, func(v int16) bool { return v != 0 }) {
				t.Fatalf("Clear後も Deltas[%d] が非ゼロ", k)
			}
		}
	})

	t.Run("異常_sampleIdxが範囲外", func(t *testing.T) {
		d, rng := newDense(t)
		rec, err := d.NewBatchRecord(2, xRows)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		_, tgt, bw := newSample(t, d, rng)
		for _, idx := range []int{-1, 2} {
			if _, err := bw(tgt, rec, idx); err == nil {
				t.Errorf("sampleIdx = %d でエラーにならない", idx)
			}
		}
	})

	t.Run("異常_別の型のBatchRecord", func(t *testing.T) {
		d, rng := newDense(t)
		_, tgt, bw := newSample(t, d, rng)
		if _, err := bw(tgt, otherRecord{}, 0); err == nil {
			t.Error("backward がエラーにならない")
		}
		if _, err := d.BatchDeltas(otherRecord{}); err == nil {
			t.Error("BatchDeltas がエラーにならない")
		}
	})
}
