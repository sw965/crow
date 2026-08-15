package main

import (
	"math"
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

// TestSatisfiesUpdateCriterionMatchesCrow は、本実験の更新判定が crow の
// binary.SatisfiesUpdateCriterion と完全に一致する事を確認する(再実装の忠実性検証)。
func TestSatisfiesUpdateCriterionMatchesCrow(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	protos := make(bitsx.Matrices, 5)
	for i := range protos {
		protos[i] = newTestMatrix(t, 2, 40, rng)
	}

	for _, margin := range []float32{0.0, 0.01, 0.1, 0.5, 1.0} {
		for trial := range 50 {
			y := newTestMatrix(t, 2, 40, rng)
			label := rng.IntN(len(protos))

			got, err := satisfiesUpdateCriterion(y, label, protos, margin)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			want, err := binary.SatisfiesUpdateCriterion(y, label, protos, margin)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if got != want {
				t.Fatalf("crowの判定と不一致 (margin = %g, trial = %d): got = %t, want = %t", margin, trial, got, want)
			}
		}
	}
}

// TestPreActivationWithBias は、活性前値が z = 2*一致数 - fanIn + bias になる事と、
// バイアス無効時にバイアスが無視される事を確認する。
func TestPreActivationWithBias(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	const wRows, wCols = 8, 64
	d, err := newDense(wRows, wCols, true, 0.5, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	for i := range d.bias {
		d.bias[i] = int32(i) - 4 // -4..3
	}
	x := newTestMatrix(t, 3, wCols, rng)

	z, yRows, yCols, err := d.preActivation(x, 0, nil)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if yRows != 3 || yCols != wRows {
		t.Fatalf("出力形状の不一致: got = (%d, %d), want = (3, %d)", yRows, yCols, wRows)
	}

	u, err := x.Dot(d.w)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	for i, count := range u {
		want := 2*count - wCols + int(d.bias[i%yCols])
		if z[i] != want {
			t.Fatalf("活性前値の不一致 (i = %d): got = %d, want = %d", i, z[i], want)
		}
	}

	// バイアス無効なら bias は無視される
	d.useBias = false
	z2, _, _, err := d.preActivation(x, 0, nil)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	for i, count := range u {
		if z2[i] != 2*count-wCols {
			t.Fatalf("バイアス無効時に bias が加算された (i = %d)", i)
		}
	}
}

// TestUpdateMovesOnlyOneOfBiasOrWeights は、依頼された学習則の核心
// 「バイアスと重みの両方は動かさない」を確認する。
func TestUpdateMovesOnlyOneOfBiasOrWeights(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	const wRows, wCols = 16, 64

	newFilledDelta := func(d *dense) *delta {
		dl := d.newDelta()
		for i := range dl.w {
			dl.w[i] = 1
		}
		for i := range dl.bias {
			dl.bias[i] = 1
		}
		return dl
	}

	t.Run("常にバイアス側_重みは動かない", func(t *testing.T) {
		d, err := newDense(wRows, wCols, true, 1.0, rng) // biasChoice = 1.0
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		wBefore := d.w.Clone()
		hBefore := make([]int8, len(d.h))
		copy(hBefore, d.h)
		biasBefore := make([]int32, len(d.bias))
		copy(biasBefore, d.bias)

		if err := d.update(newFilledDelta(d), 1.0, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}

		if !d.w.Equal(wBefore) {
			t.Error("バイアス側が選ばれたのに可視重みが変化した")
		}
		for i, v := range d.h {
			if v != hBefore[i] {
				t.Fatalf("バイアス側が選ばれたのに隠れ重みが変化した (i = %d)", i)
			}
		}
		for i, v := range d.bias {
			if v != biasBefore[i]+1 {
				t.Fatalf("バイアスが+1されていない (i = %d): got = %d, want = %d", i, v, biasBefore[i]+1)
			}
		}
	})

	t.Run("常に重み側_バイアスは動かない", func(t *testing.T) {
		d, err := newDense(wRows, wCols, true, 0.0, rng) // biasChoice = 0.0
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		biasBefore := make([]int32, len(d.bias))
		copy(biasBefore, d.bias)
		hBefore := make([]int8, len(d.h))
		copy(hBefore, d.h)

		if err := d.update(newFilledDelta(d), 1.0, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}

		for i, v := range d.bias {
			if v != biasBefore[i] {
				t.Fatalf("重み側が選ばれたのにバイアスが変化した (i = %d)", i)
			}
		}
		changed := false
		for i, v := range d.h {
			if v != hBefore[i] {
				changed = true
				break
			}
			_ = i
		}
		if !changed {
			t.Error("重み側が選ばれたのに隠れ重みが変化しなかった")
		}
	})
}

// TestUpdateKeepsVisibleWeightConsistentWithHidden は、H の符号と W のビットが
// 更新後も一致し続ける事(crowと同じ不変条件)を確認する。
func TestUpdateKeepsVisibleWeightConsistentWithHidden(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const wRows, wCols = 8, 64
	d, err := newDense(wRows, wCols, false, 0, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	for range 20 {
		dl := d.newDelta()
		for i := range dl.w {
			dl.w[i] = int16(rng.IntN(3) - 1)
		}
		if err := d.update(dl, 1.0, rng); err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		for row := range wRows {
			for col := range wCols {
				bit, err := d.w.Bit(row, col)
				if err != nil {
					t.Fatalf("予期せぬエラー: %v", err)
				}
				want := uint64(0)
				if d.h[row*wCols+col] >= 0 {
					want = 1
				}
				if bit != want {
					t.Fatalf("Hの符号とWのビットが不一致 (row = %d, col = %d)", row, col)
				}
				// 転置も同期している事
				bitT, err := d.wt.Bit(col, row)
				if err != nil {
					t.Fatalf("予期せぬエラー: %v", err)
				}
				if bitT != bit {
					t.Fatalf("転置行列が同期していない (row = %d, col = %d)", row, col)
				}
			}
		}
	}
}

// TestApplyCellMaskPreservesPairwiseDistance は、セルマスクが見本同士の
// ハミング距離を変えない(=符号の距離幾何を保つ対称変換である)事を確認する。
// B-9c の「効くのは点灯率を均すからではない」という解釈の前提になる性質。
func TestApplyCellMaskPreservesPairwiseDistance(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	protos, err := bitsx.NewThermometerMatrices(11, 4, 32)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	before := make([][]int, len(protos))
	for i := range protos {
		before[i] = make([]int, len(protos))
		for j := range protos {
			d, err := protos[i].HammingDistance(protos[j])
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			before[i][j] = d
		}
	}

	if err := applyCellMask(protos, rng); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	for i := range protos {
		for j := range protos {
			d, err := protos[i].HammingDistance(protos[j])
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if d != before[i][j] {
				t.Fatalf("セルマスクで見本間距離が変化した (%d, %d): got = %d, want = %d", i, j, d, before[i][j])
			}
		}
	}
}

// TestWeightedAverage は加重平均復号の基本性質を確認する。
func TestWeightedAverage(t *testing.T) {
	values := []float64{0, 10, 20, 30}

	t.Run("温度が非常に小さいと最大ロジットの値へ収束する", func(t *testing.T) {
		got := weightedAverage([]int{0, 5, 100, 3}, values, 0.01)
		if math.Abs(got-20) > 1e-6 {
			t.Errorf("値の不一致: got = %f, want = 20", got)
		}
	})

	t.Run("温度が非常に大きいと全値の平均へ近づく", func(t *testing.T) {
		got := weightedAverage([]int{0, 5, 100, 3}, values, 1e9)
		want := (0.0 + 10 + 20 + 30) / 4
		if math.Abs(got-want) > 1e-3 {
			t.Errorf("値の不一致: got = %f, want = %f", got, want)
		}
	})

	t.Run("常に値域内に収まる", func(t *testing.T) {
		rng := rand.New(rand.NewPCG(11, 12))
		for range 200 {
			logits := make([]int, len(values))
			for i := range logits {
				logits[i] = rng.IntN(200)
			}
			got := weightedAverage(logits, values, float64(1+rng.IntN(300)))
			if got < values[0] || got > values[len(values)-1] {
				t.Fatalf("値域外: got = %f", got)
			}
		}
	})
}

// TestRegressionDataMatchesLevelTrend は、合成回帰データが
// 「レベルが上がるほど方向行列 D との一致が増える」性質を持つ事を確認する。
func TestRegressionDataMatchesLevelTrend(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))
	const levels, rows, cols = 11, 8, 64
	d := newTestMatrix(t, rows, cols, rng)

	// 各レベル200件ずつ生成し、平均一致率が単調に増えることを見る
	const perLevel = 200
	xs, lv, err := newRegressionData(levels, levels*perLevel, rows, cols, d, rng)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	total := rows * cols
	sums := make([]float64, levels)
	counts := make([]int, levels)
	for i, x := range xs {
		hd, err := x.HammingDistance(d)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		sums[lv[i]] += float64(total-hd) / float64(total)
		counts[lv[i]]++
	}

	first := sums[0] / float64(counts[0])
	last := sums[levels-1] / float64(counts[levels-1])
	// q(0) = 0.35, q(n-1) = 0.65 なので、両端で明確に差が出る
	if math.Abs(first-0.35) > 0.05 {
		t.Errorf("レベル0の一致率が想定外: got = %.3f, want = 0.35 前後", first)
	}
	if math.Abs(last-0.65) > 0.05 {
		t.Errorf("最終レベルの一致率が想定外: got = %.3f, want = 0.65 前後", last)
	}
	if last <= first {
		t.Errorf("レベルとともに一致率が増えていない: %.3f -> %.3f", first, last)
	}
}

// ---------------------------------------------------------------------------
// crow 本体との差分テスト
// ---------------------------------------------------------------------------

// newPairedDenses は、crow の Dense と本実験の dense を「同じ W / WT / H」で用意する。
func newPairedDenses(t *testing.T, wRows, wCols int, seed uint64) (*binary.Dense, *dense) {
	t.Helper()
	crowDense, err := binary.NewDense(wRows, wCols, rand.New(rand.NewPCG(seed, seed+1)))
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	// 共有ハイパーパラメータ(ノイズ0で決定的にする)
	seq := binary.Sequence{crowDense}
	ctx := binary.NewSharedHyperparameters()
	ctx.NoiseStdScale = 0
	if err := seq.SetSharedHyperparameters(&ctx); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	mine, err := newDense(wRows, wCols, false, 0, rand.New(rand.NewPCG(seed, seed+1)))
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	// W / WT / H を crow 側からコピーして完全に揃える
	mine.w = crowDense.W.Clone()
	mine.wt = crowDense.WT.Clone()
	mine.h = make([]int8, len(crowDense.H))
	copy(mine.h, crowDense.H)
	mine.groupSize = ctx.GroupSize
	mine.gateScale = ctx.GateDropThresholdScale
	return crowDense, mine
}

func assertSameMatrix(t *testing.T, label string, got, want *bitsx.Matrix) {
	t.Helper()
	if !got.Equal(want) {
		t.Fatalf("%s が crow と一致しない", label)
	}
}

// TestPredictMatchesCrow は、推論経路(バイアス無し)が crow と bit 単位で一致する事を確認する。
func TestPredictMatchesCrow(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	for _, shape := range [][2]int{{8, 64}, {16, 100}, {33, 37}} {
		crowDense, mine := newPairedDenses(t, shape[0], shape[1], 100+uint64(shape[0]))
		for range 10 {
			x := newTestMatrix(t, 3, shape[1], rng)
			want, err := crowDense.Predict(x)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			got, err := mine.predict(x)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			assertSameMatrix(t, "Predict出力", got, want)
		}
	}
}

// TestForwardBackwardDeltaMatchesCrow は、順伝播の出力と逆伝播で溜まるデルタが
// crow と完全に一致する事を確認する(ノイズ0で決定的に比較)。
func TestForwardBackwardDeltaMatchesCrow(t *testing.T) {
	rng := rand.New(rand.NewPCG(23, 24))
	for _, shape := range [][2]int{{8, 64}, {16, 100}, {33, 37}} {
		crowDense, mine := newPairedDenses(t, shape[0], shape[1], 200+uint64(shape[0]))

		for range 10 {
			x := newTestMatrix(t, 3, shape[1], rng)
			target := newTestMatrix(t, 3, shape[0], rng)

			crowY, crowBw, err := crowDense.Forward(x, rand.New(rand.NewPCG(1, 1)))
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			myY, myBw, err := mine.forward(x, 0, nil)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			assertSameMatrix(t, "Forward出力", myY, crowY)

			crowDeltas := crowDense.NewZerosDeltas()
			crowNext, err := crowBw(target, crowDeltas)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			myDelta := mine.newDelta()
			myNext, err := myBw(target, myDelta)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			assertSameMatrix(t, "逆伝播で返る希望活性", myNext, crowNext)

			if len(crowDeltas[0]) != len(myDelta.w) {
				t.Fatalf("デルタ長の不一致: got = %d, want = %d", len(myDelta.w), len(crowDeltas[0]))
			}
			for i, v := range crowDeltas[0] {
				if myDelta.w[i] != v {
					t.Fatalf("デルタが crow と不一致 (i = %d): got = %d, want = %d", i, myDelta.w[i], v)
				}
			}
		}
	}
}

// TestUpdateMatchesCrow は、同じデルタ・同じ乱数列で Update した後の H と W が
// crow と完全に一致する事を確認する。
// biasChoice=0 のとき本実験の update は乱数を余分に消費しないので、1ステップ比較できる。
func TestUpdateMatchesCrow(t *testing.T) {
	rng := rand.New(rand.NewPCG(25, 26))
	for _, shape := range [][2]int{{8, 64}, {16, 100}, {33, 37}} {
		crowDense, mine := newPairedDenses(t, shape[0], shape[1], 300+uint64(shape[0]))

		for step := range 10 {
			crowDeltas := crowDense.NewZerosDeltas()
			myDelta := mine.newDelta()
			for i := range crowDeltas[0] {
				v := int16(rng.IntN(5) - 2)
				crowDeltas[0][i] = v
				myDelta.w[i] = v
			}

			seed := uint64(1000 + step)
			if err := crowDense.Update(crowDeltas, 0.5, rand.New(rand.NewPCG(seed, seed+1))); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
			if err := mine.update(myDelta, 0.5, rand.New(rand.NewPCG(seed, seed+1))); err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}

			assertSameMatrix(t, "更新後のW", mine.w, crowDense.W)
			assertSameMatrix(t, "更新後のWT", mine.wt, crowDense.WT)
			for i, v := range crowDense.H {
				if mine.h[i] != v {
					t.Fatalf("更新後のHが crow と不一致 (step = %d, i = %d): got = %d, want = %d",
						step, i, mine.h[i], v)
				}
			}
		}
	}
}
