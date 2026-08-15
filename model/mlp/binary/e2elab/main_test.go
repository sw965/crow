package main

import (
	"math/rand/v2"
	"testing"

	"github.com/sw965/crow/model/mlp/binary"
	"github.com/sw965/omw/mathx/bitsx"
)

func newTestMatrix(t *testing.T, rows, cols int, rng *rand.Rand) *bitsx.Matrix {
	t.Helper()
	m, err := bitsx.NewRandMatrix(rows, cols, 0, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	return m
}

func newTestClassifier(t *testing.T, rows, cols int, learnable bool, rng *rand.Rand) *classifier {
	t.Helper()
	protos := make(bitsx.Matrices, numClasses)
	for i := range protos {
		protos[i] = newTestMatrix(t, rows, cols, rng)
	}
	c, err := newClassifier(protos, learnable)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	return c
}

func bitAt(t *testing.T, m *bitsx.Matrix, r, c int) uint64 {
	t.Helper()
	b, err := m.Bit(r, c)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	return b
}

// TestClassifierLogits は、ロジットが「一致ビット数 = 総ビット - ハミング距離」である事を確認する。
func TestClassifierLogits(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	c := newTestClassifier(t, 2, 100, false, rng)
	y := newTestMatrix(t, 2, 100, rng)

	got, err := c.logits(y)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	for cls := range numClasses {
		hd, err := y.HammingDistance(c.q[cls])
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		want := c.totalBits() - hd
		if got[cls] != want {
			t.Fatalf("ロジットの不一致 (cls = %d): got = %d, want = %d", cls, got[cls], want)
		}
	}
}

// TestClassifierInitialisedFromPrototypes は、H の符号が初期プロトタイプと一致する事を確認する。
func TestClassifierInitialisedFromPrototypes(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	const rows, cols = 2, 100
	c := newTestClassifier(t, rows, cols, true, rng)
	for cls := range numClasses {
		idx := 0
		for r := range rows {
			for col := range cols {
				bit := bitAt(t, c.q[cls], r, col)
				h := c.h[cls][idx]
				wantPositive := bit == 1
				if (h >= 0) != wantPositive {
					t.Fatalf("Hの符号がプロトタイプと不一致 (cls = %d, idx = %d): bit = %d, h = %d", cls, idx, bit, h)
				}
				idx++
			}
		}
	}
}

func TestRival(t *testing.T) {
	logits := []int{5, 30, 12, 30, 1, 0, 0, 0, 0, 0}
	// label=1 のとき、残りの最大は index 3
	if got := rival(logits, 1); got != 3 {
		t.Errorf("競合クラスの不一致: got = %d, want = 3", got)
	}
	// label=3 のとき、残りの最大は index 1
	if got := rival(logits, 3); got != 1 {
		t.Errorf("競合クラスの不一致: got = %d, want = 1", got)
	}
}

// TestDesiredActivationDiff は diff 目標の構成を確認する。
// 正解行と競合行が食い違うビットは正解行に従い、一致するビットは現在の活性のままになる。
func TestDesiredActivationDiff(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	const rows, cols = 2, 100
	c := newTestClassifier(t, rows, cols, false, rng)
	y := newTestMatrix(t, rows, cols, rng)

	const label, riv = 2, 7
	rngSel := rand.New(rand.NewPCG(99, 100))
	sel, err := c.selectionMask(label, riv, "diff", 0, rngSel)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	got, err := c.targetFrom(y, label, sel)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	sameCount, diffCount := 0, 0
	for r := range rows {
		for col := range cols {
			qy := bitAt(t, c.q[label], r, col)
			qr := bitAt(t, c.q[riv], r, col)
			want := qy
			if qy == qr {
				want = bitAt(t, y, r, col)
				sameCount++
			} else {
				diffCount++
			}
			if g := bitAt(t, got, r, col); g != want {
				t.Fatalf("目標ビットの不一致 (r = %d, col = %d): got = %d, want = %d", r, col, g, want)
			}
		}
	}
	if sameCount == 0 || diffCount == 0 {
		t.Fatalf("検証が退化している: 一致 %d / 不一致 %d", sameCount, diffCount)
	}
}

// TestDesiredActivationProto は proto 目標が正解プロトタイプそのものである事を確認する。
func TestDesiredActivationProto(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	c := newTestClassifier(t, 2, 100, false, rng)
	y := newTestMatrix(t, 2, 100, rng)
	rngSel := rand.New(rand.NewPCG(101, 102))
	sel, err := c.selectionMask(3, 5, "proto", 0, rngSel)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	got, err := c.targetFrom(y, 3, sel)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if !got.Equal(c.q[3]) {
		t.Error("proto目標が正解プロトタイプと一致しない")
	}
}

// TestCareMask は、careMask が「正解行と競合行が食い違うビット」である事を確認する。
func TestCareMask(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	const rows, cols = 2, 100
	c := newTestClassifier(t, rows, cols, false, rng)
	mask, err := c.careMask(1, 4)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	for r := range rows {
		for col := range cols {
			want := bitAt(t, c.q[1], r, col) ^ bitAt(t, c.q[4], r, col)
			if got := bitAt(t, mask, r, col); got != want {
				t.Fatalf("careMaskの不一致 (r = %d, col = %d)", r, col)
			}
		}
	}
}

// TestClassifierUpdateDirection は、学習可能な分類器が
// 正解クラス行を活性へ近づけ、競合クラス行を遠ざける事を確認する。
func TestClassifierUpdateDirection(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	const rows, cols = 2, 100
	c := newTestClassifier(t, rows, cols, true, rng)
	y := newTestMatrix(t, rows, cols, rng)

	const label, riv = 0, 6
	beforeLabel, err := y.HammingDistance(c.q[label])
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	beforeRival, err := y.HammingDistance(c.q[riv])
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	// hInitAbs 回以上繰り返せば、符号が変わるべきビットは反転する
	for range hInitAbs + 2 {
		dl := make([][]int16, numClasses)
		for i := range dl {
			dl[i] = make([]int16, c.totalBits())
		}
		if err := c.accumulate(y, label, riv, dl); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		for i := range dl {
			for j, v := range dl[i] {
				if v > 0 {
					dl[i][j] = 1
				} else if v < 0 {
					dl[i][j] = -1
				}
			}
		}
		if err := c.update(dl, 1.0, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
	}

	afterLabel, err := y.HammingDistance(c.q[label])
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	afterRival, err := y.HammingDistance(c.q[riv])
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	if afterLabel != 0 {
		t.Errorf("正解クラス行が活性へ収束していない: %d -> %d", beforeLabel, afterLabel)
	}
	if afterRival != c.totalBits() {
		t.Errorf("競合クラス行が活性から最も遠ざかっていない: %d -> %d (総ビット %d)",
			beforeRival, afterRival, c.totalBits())
	}
}

// TestClassifierNotLearnableStaysFixed は、learnable=false なら分類器が動かない事を確認する。
func TestClassifierNotLearnableStaysFixed(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))
	c := newTestClassifier(t, 2, 100, false, rng)
	y := newTestMatrix(t, 2, 100, rng)
	before := make(bitsx.Matrices, numClasses)
	for i := range c.q {
		before[i] = c.q[i].Clone()
	}

	dl := make([][]int16, numClasses)
	for i := range dl {
		dl[i] = make([]int16, c.totalBits())
		for j := range dl[i] {
			dl[i][j] = 1
		}
	}
	if err := c.accumulate(y, 0, 1, dl); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if err := c.update(dl, 1.0, rng); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	for i := range c.q {
		if !c.q[i].Equal(before[i]) {
			t.Fatalf("learnable=false なのに分類器行が変化した (cls = %d)", i)
		}
	}
}

// TestSatisfiesUpdateCriterionMatchesCrow は、更新判定が crow の
// binary.SatisfiesUpdateCriterion と一致する事を確認する。
// 本実験は既に求めたロジットで判定するので、同じ入力から両者を突き合わせる。
func TestSatisfiesUpdateCriterionMatchesCrow(t *testing.T) {
	rng := rand.New(rand.NewPCG(15, 16))
	const rows, cols = 2, 100
	c := newTestClassifier(t, rows, cols, false, rng)

	for _, margin := range []float32{0.0, 0.01, 0.1, 0.5, 1.0} {
		for range 50 {
			y := newTestMatrix(t, rows, cols, rng)
			label := rng.IntN(numClasses)

			lg, err := c.logits(y)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			got := satisfiesUpdateCriterion(lg, label, rival(lg, label), c.totalBits(), margin)

			want, err := binary.SatisfiesUpdateCriterion(y, label, c.q, margin)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if got != want {
				t.Fatalf("crowの判定と不一致 (margin = %g): got = %t, want = %t", margin, got, want)
			}
		}
	}
}

// TestSelectionMaskRandDiff は、randdiff が「正解行と競合行が食い違うビット」から
// ちょうど n 個を選ぶ事を確認する。
func TestSelectionMaskRandDiff(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	const rows, cols = 2, 100
	c := newTestClassifier(t, rows, cols, false, rng)
	const label, riv = 1, 8

	pool, err := c.q[label].Xor(c.q[riv])
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	poolCount := pool.OnesCount()
	if poolCount == 0 {
		t.Fatal("母集団が空で検証にならない")
	}

	for _, n := range []int{1, 8, 32, poolCount, poolCount + 100} {
		sel, err := c.selectionMask(label, riv, "randdiff", n, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		want := min(n, poolCount)
		if got := sel.OnesCount(); got != want {
			t.Fatalf("選択ビット数の不一致 (n = %d): got = %d, want = %d", n, got, want)
		}
		// 選ばれたビットは必ず母集団の部分集合
		and, err := sel.And(pool)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !and.Equal(sel) {
			t.Fatalf("母集団の外のビットが選ばれた (n = %d)", n)
		}
	}
}

// TestSelectionMaskRandAll は、randall が全ビットから n 個選ぶ事を確認する。
func TestSelectionMaskRandAll(t *testing.T) {
	rng := rand.New(rand.NewPCG(23, 24))
	const rows, cols = 2, 100
	c := newTestClassifier(t, rows, cols, false, rng)
	for _, n := range []int{1, 16, 64} {
		sel, err := c.selectionMask(0, 1, "randall", n, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if got := sel.OnesCount(); got != n {
			t.Fatalf("選択ビット数の不一致 (n = %d): got = %d", n, got)
		}
	}
}

// TestTargetFromLeavesUnselectedBitsAsActivation は、選ばれなかったビットが
// 現在の活性のままである事(=BEPの逆伝播で更新対象に選ばれない事)を確認する。
func TestTargetFromLeavesUnselectedBitsAsActivation(t *testing.T) {
	rng := rand.New(rand.NewPCG(25, 26))
	const rows, cols = 2, 100
	c := newTestClassifier(t, rows, cols, false, rng)
	y := newTestMatrix(t, rows, cols, rng)

	sel, err := c.selectionMask(3, 6, "randdiff", 16, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	got, err := c.targetFrom(y, 3, sel)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	for r := range rows {
		for col := range cols {
			if bitAt(t, sel, r, col) == 1 {
				if bitAt(t, got, r, col) != bitAt(t, c.q[3], r, col) {
					t.Fatalf("選ばれたビットが正解プロトタイプと違う (r = %d, col = %d)", r, col)
				}
			} else {
				if bitAt(t, got, r, col) != bitAt(t, y, r, col) {
					t.Fatalf("選ばれなかったビットが活性と違う (r = %d, col = %d)", r, col)
				}
			}
		}
	}
}
