// e2elab: 固定プロトタイプからバックボーンへ送る学習信号を変える実験。
//
// 動機: 現在の BEP は最終活性へ固定プロトタイプ P_y の完全再現を要求する。
// しかし分類に本当に必要なのは P_y·a > P_c·a (c≠y) だけであり、
// BEP は分類問題を、より厳しい「符号再現問題」へ変換してしまっている。
// 後付けの整数読み出しはこのずれを外側で吸収するが、
// END-TO-END の美しさ(最後まで二値ネット自身が表現を作る)は失われる。
//
// そこで2つの軸を切り分けて測る。
//
//	(a) 目標を指定するビットの選び方 (-target と -n / -q)
//	    proto            : 全ビット(従来のBEP)
//	    diff             : 正解 P_y と最大競合 P_r が食い違うビット全部
//	    randdiff -n N    : 上記から無作為に N 個
//	    randall  -n N    : 全ビットから無作為に N 個(対照)
//	  選ばれなかったビットには現在の活性をそのまま置く。
//	  BEP の逆伝播は目標と活性の不一致だけを更新対象に選ぶので、
//	  これが「そのビットには誤差信号を送らない」の実装になる。
//	  -hard を付けると、選ばれなかったビットをゲートからも外し、
//	  後方射影 sign(W^T (g ⊙ t)) にも寄与させない。
//
//	(b) 分類器の学習 (-learnq)
//	    Q = sign(H_Q) とし、H_Q を BEP と同じ整数隠れ重みとして学習する。
//	    初期値は ETF プロトタイプ。正解クラス行を現在の活性へ近づけ、
//	    最大競合クラス行を遠ざける。-learnq=false なら Q は一切動かない。
//
// 推論経路は最後まで二値活性・二値可視重み・XNOR・popcount・整数比較だけで、
// 整数読み出しは存在しない。
//
// 結果はどの案も元の BEP を上回らなかった。詳細と原因の診断は REPORT.md を参照。
//
// BEP の順伝播・逆伝播・更新は ../layer.go ../train.go と同じ手順を bitsx の
// 公開APIで再実装したもので、../biaslab と同じ土台を使っている
// (crow 本体との1ステップ差分テストは ../biaslab/main_test.go にある)。
// crow / omw のライブラリコードは変更していない。
//
// 実行例:
//
//	go run . -target proto                 # 従来のBEP
//	go run . -target diff                  # 競合差分の全ビット
//	go run . -target randdiff -n 128       # 競合差分から128個
//	go run . -target randall  -n 128       # 全ビットから128個(対照)
//	go run . -target proto -learnq         # 分類器も学習する
package main

import (
	"cmp"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"time"

	"github.com/sw965/crow/dataset"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
)

const (
	hInitAbs   = 4 // ../layer.go と同じ
	numClasses = 10
)

// ---------------------------------------------------------------------------
// デルタ
// ---------------------------------------------------------------------------

type delta struct {
	w []int16 // wRows * wCols
}

func newDelta(wRows, wCols int) *delta { return &delta{w: make([]int16, wRows*wCols)} }

func (d *delta) clear() { clear(d.w) }

func (d *delta) add(other *delta) {
	for i, v := range other.w {
		d.w[i] += v
	}
}

func (d *delta) sign() {
	for i, v := range d.w {
		d.w[i] = int16(cmp.Compare(v, 0))
	}
}

// ---------------------------------------------------------------------------
// Dense層(../layer.go の再実装)
// ---------------------------------------------------------------------------

type dense struct {
	w  *bitsx.Matrix
	wt *bitsx.Matrix
	h  []int8

	gateBase  int
	noiseStd  float32
	gateScale float32
	groupSize int
}

func newDense(wRows, wCols int, rng *rand.Rand) (*dense, error) {
	w, err := bitsx.NewRandMatrix(wRows, wCols, 0, rng)
	if err != nil {
		return nil, err
	}
	wt, err := w.Transpose()
	if err != nil {
		return nil, err
	}
	h := make([]int8, wRows*wCols)
	if err := w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		word, err := w.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		hWord := h[ctx.GlobalStart:ctx.GlobalEnd]
		return ctx.ScanBits(func(i, col, colT int) error {
			if word>>uint64(i)&1 == 1 {
				hWord[i] = hInitAbs
			} else {
				hWord[i] = -hInitAbs
			}
			return nil
		})
	}); err != nil {
		return nil, err
	}
	noiseStdBase := float32(math.Sqrt(float64(wCols)))
	return &dense{
		w: w, wt: wt, h: h,
		gateBase: int(noiseStdBase), noiseStd: noiseStdBase,
		gateScale: 1.0, groupSize: 4,
	}, nil
}

func (d *dense) outputShape(xRows, xCols int) (int, int, error) {
	if xCols != d.w.Cols() {
		return 0, 0, fmt.Errorf("入力の列数が不一致: xCols = %d, W.Cols = %d", xCols, d.w.Cols())
	}
	return xRows, d.w.Rows(), nil
}

func (d *dense) newDelta() *delta { return newDelta(d.w.Rows(), d.w.Cols()) }

func (d *dense) preActivation(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) ([]int, int, int, error) {
	u, err := x.Dot(d.w)
	if err != nil {
		return nil, 0, 0, err
	}
	fanIn := d.w.Cols()
	z := make([]int, len(u))
	std := noiseScale * d.noiseStd
	for i, count := range u {
		zi := 2*count - fanIn
		if std > 0 {
			noise, err := randx.IntNorm(-fanIn, fanIn, 0, std, rng)
			if err != nil {
				return nil, 0, 0, err
			}
			zi += noise
		}
		z[i] = zi
	}
	return z, x.Rows(), d.w.Rows(), nil
}

// backward は希望活性 t を受け取り、前段への希望活性を返す。
// care が非nilなら、そのビットが0の位置はゲートから外し、
// 後方射影 sign(W^T (g ⊙ t)) に寄与させない。
// 「その位置は指定なし」を、局所の更新対象からだけでなく
// 前段への信号からも取り除くために使う。
type backward func(t *bitsx.Matrix, care *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error)

func (d *dense) forward(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) (*bitsx.Matrix, backward, error) {
	z, yRows, yCols, err := d.preActivation(x, noiseScale, rng)
	if err != nil {
		return nil, nil, err
	}
	y, err := bitsx.NewSignMatrix(yRows, yCols, z)
	if err != nil {
		return nil, nil, err
	}

	bw := func(t *bitsx.Matrix, care *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error) {
		if err := t.ValidateSameShape(y); err != nil {
			return nil, err
		}
		keepGate, err := bitsx.NewZerosMatrix(yRows, yCols)
		if err != nil {
			return nil, err
		}
		gate := int(d.gateScale * float32(d.gateBase))

		err = t.ScanRowsWord(nil, func(tCtx bitsx.MatrixWordContext) error {
			zWord := z[tCtx.GlobalStart:tCtx.GlobalEnd]
			type wordMismatch struct {
				absZi int
				tBit  uint64
				col   int
			}
			mismatches := make([]wordMismatch, 0, 64)
			tWord, err := t.Word(tCtx.WordIndex)
			if err != nil {
				return err
			}
			var keepGateWord uint64

			if err := tCtx.ScanBits(func(i, col, colT int) error {
				zi := zWord[i]
				absZi := zi
				if absZi < 0 {
					absZi = -absZi
				}
				if absZi <= gate {
					keepGateWord |= 1 << uint64(i)
				}
				tBit := tWord >> uint64(i) & 1
				yBit := uint64(0)
				if zi >= 0 {
					yBit = 1
				}
				if tBit != yBit {
					mismatches = append(mismatches, wordMismatch{absZi: absZi, tBit: tBit, col: col})
				}
				return nil
			}); err != nil {
				return err
			}
			if care != nil {
				careWord, err := care.Word(tCtx.WordIndex)
				if err != nil {
					return err
				}
				keepGateWord &= careWord
			}
			if err := keepGate.SetWord(tCtx.WordIndex, keepGateWord); err != nil {
				return err
			}
			slices.SortFunc(mismatches, func(a, b wordMismatch) int {
				return cmp.Compare(a.absZi, b.absZi)
			})
			updateK := min(max(len(zWord)/d.groupSize, 1), len(mismatches))

			for _, mm := range mismatches[:updateK] {
				deltaRow := dl.w[mm.col*d.w.Cols() : (mm.col+1)*d.w.Cols()]
				if err := x.ScanRowsWord([]int{tCtx.Row}, func(xCtx bitsx.MatrixWordContext) error {
					xWord, err := x.Word(xCtx.WordIndex)
					if err != nil {
						return err
					}
					dw := deltaRow[xCtx.ColStart:xCtx.ColEnd]
					for b := range dw {
						xBit := xWord >> uint(b) & 1
						dw[b] += int16(1 - 2*int(xBit^mm.tBit))
					}
					return nil
				}); err != nil {
					return err
				}
			}
			return nil
		})
		if err != nil {
			return nil, err
		}

		rawNextT, err := d.wt.DotTernary(t, keepGate)
		if err != nil {
			return nil, err
		}
		nextT, err := bitsx.NewZerosMatrix(yRows, d.w.Cols())
		if err != nil {
			return nil, err
		}
		if err := nextT.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
			var word uint64
			if err := ctx.ScanBits(func(i, col, colT int) error {
				if rawNextT[colT] >= 0 {
					word |= 1 << uint(i)
				}
				return nil
			}); err != nil {
				return err
			}
			return nextT.SetWord(ctx.WordIndex, word)
		}); err != nil {
			return nil, err
		}
		return nextT, nil
	}
	return y, bw, nil
}

func (d *dense) predict(x *bitsx.Matrix) (*bitsx.Matrix, error) {
	z, yRows, yCols, err := d.preActivation(x, 0, nil)
	if err != nil {
		return nil, err
	}
	return bitsx.NewSignMatrix(yRows, yCols, z)
}

func (d *dense) update(dl *delta, lr float32, rng *rand.Rand) error {
	return d.w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		hWord := d.h[ctx.GlobalStart:ctx.GlobalEnd]
		dWord := dl.w[ctx.GlobalStart:ctx.GlobalEnd]
		var flips uint64
		if err := ctx.ScanBits(func(i, col, colT int) error {
			if rng.Float32() > lr {
				return nil
			}
			old := hWord[i]
			clipped := int8(max(math.MinInt8, min(int(old)+int(dWord[i]), math.MaxInt8)))
			hWord[i] = clipped
			if (old >= 0) != (clipped >= 0) {
				flips |= 1 << uint64(i)
				return d.wt.Toggle(col, ctx.Row)
			}
			return nil
		}); err != nil {
			return err
		}
		old, err := d.w.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		return d.w.SetWord(ctx.WordIndex, old^flips)
	})
}

// ---------------------------------------------------------------------------
// 出力分類器: 学習可能な二値層
// ---------------------------------------------------------------------------

// classifier はクラスごとに二値の重み行 Q_c を持つ出力層。
// Q_c = sign(H_c) で、H_c は BEP の隠れ重みと同じ整数(int8)。
// ロジットは一致ビット数(= 総ビット数 - ハミング距離)で、XNOR/popcount で求まる。
// learnable=false なら固定プロトタイプ(従来のBEP)と同じ。
type classifier struct {
	q         bitsx.Matrices // numClasses 個、各 yRows x yCols
	h         [][]int8       // numClasses 個、各 yRows*yCols
	learnable bool
	rows      int
	cols      int
}

func newClassifier(protos bitsx.Matrices, learnable bool) (*classifier, error) {
	if len(protos) != numClasses {
		return nil, fmt.Errorf("プロトタイプ数が不正: %d", len(protos))
	}
	c := &classifier{
		q: protos, learnable: learnable,
		rows: protos[0].Rows(), cols: protos[0].Cols(),
	}
	c.h = make([][]int8, numClasses)
	for cls := range numClasses {
		n := c.rows * c.cols
		hc := make([]int8, n)
		if err := protos[cls].ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
			word, err := protos[cls].Word(ctx.WordIndex)
			if err != nil {
				return err
			}
			hw := hc[ctx.GlobalStart:ctx.GlobalEnd]
			return ctx.ScanBits(func(i, col, colT int) error {
				if word>>uint64(i)&1 == 1 {
					hw[i] = hInitAbs
				} else {
					hw[i] = -hInitAbs
				}
				return nil
			})
		}); err != nil {
			return nil, err
		}
		c.h[cls] = hc
	}
	return c, nil
}

func (c *classifier) totalBits() int { return c.rows * c.cols }

// logits は各クラスの一致ビット数を返す。
func (c *classifier) logits(y *bitsx.Matrix) ([]int, error) {
	total := c.totalBits()
	out := make([]int, numClasses)
	for i, q := range c.q {
		hd, err := y.HammingDistance(q)
		if err != nil {
			return nil, err
		}
		out[i] = total - hd
	}
	return out, nil
}

// rival は正解以外で最大のロジットを持つクラスを返す。
func rival(logits []int, label int) int {
	best, bestVal := -1, math.MinInt
	for c, v := range logits {
		if c == label {
			continue
		}
		if v > bestVal {
			bestVal, best = v, c
		}
	}
	return best
}

// selectionMask は、今回の更新で「目標を指定する」ビットを選んだ行列を返す。
// 1 のビットだけが目標を持ち、0 のビットには誤差信号を送らない。
//
//	proto    : 全ビット(従来のBEP)
//	diff     : 正解行と競合行が食い違うビット全部
//	randall  : 全ビットから無作為に n 個
//	randdiff : 正解行と競合行が食い違うビットから無作為に n 個
//
// n <= 0 なら絞り込まず、その母集団を全部使う。
func (c *classifier) selectionMask(label, riv int, mode string, n int, rng *rand.Rand) (*bitsx.Matrix, error) {
	switch mode {
	case "proto":
		return bitsx.NewOnesMatrix(c.rows, c.cols)
	case "diff":
		if riv < 0 {
			return bitsx.NewOnesMatrix(c.rows, c.cols)
		}
		return c.q[label].Xor(c.q[riv])
	case "randall", "randdiff":
		var pool *bitsx.Matrix
		var err error
		if mode == "randall" {
			pool, err = bitsx.NewOnesMatrix(c.rows, c.cols)
		} else if riv < 0 {
			pool, err = bitsx.NewOnesMatrix(c.rows, c.cols)
		} else {
			pool, err = c.q[label].Xor(c.q[riv])
		}
		if err != nil {
			return nil, err
		}
		if n <= 0 {
			return pool, nil
		}
		return sampleBits(pool, n, rng)
	}
	return nil, fmt.Errorf("目標の作り方が不正: %s", mode)
}

// sampleBits は、pool の立っているビットから重複なしで n 個を選んだ行列を返す。
// n が母集団以上なら pool をそのまま返す。
func sampleBits(pool *bitsx.Matrix, n int, rng *rand.Rand) (*bitsx.Matrix, error) {
	idxs := make([]int, 0, pool.Rows()*pool.Cols())
	if err := pool.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		word, err := pool.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		return ctx.ScanBits(func(i, col, colT int) error {
			if word>>uint64(i)&1 == 1 {
				idxs = append(idxs, ctx.Row*pool.Cols()+col)
			}
			return nil
		})
	}); err != nil {
		return nil, err
	}
	if n >= len(idxs) {
		return pool, nil
	}
	// 部分 Fisher-Yates で先頭 n 個を無作為に決める
	for i := range n {
		j := i + rng.IntN(len(idxs)-i)
		idxs[i], idxs[j] = idxs[j], idxs[i]
	}
	sel, err := bitsx.NewZerosMatrix(pool.Rows(), pool.Cols())
	if err != nil {
		return nil, err
	}
	for _, idx := range idxs[:n] {
		if err := sel.Set(idx/pool.Cols(), idx%pool.Cols()); err != nil {
			return nil, err
		}
	}
	return sel, nil
}

// targetFrom は、選ばれたビットには正解行の値を、選ばれなかったビットには
// 現在の活性をそのまま置いた目標を作る。
// 現在の活性と一致するビットは BEP の逆伝播で更新対象に選ばれないため、
// これが「誤差信号を送らない」の実装になる。
func (c *classifier) targetFrom(y *bitsx.Matrix, label int, sel *bitsx.Matrix) (*bitsx.Matrix, error) {
	qy := c.q[label]
	t, err := bitsx.NewZerosMatrix(c.rows, c.cols)
	if err != nil {
		return nil, err
	}
	if err := t.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		yw, err := y.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		qw, err := qy.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		sw, err := sel.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		return t.SetWord(ctx.WordIndex, (qw&sw)|(yw&^sw))
	}); err != nil {
		return nil, err
	}
	return t, nil
}

// careMask は「指定あり」のビット(正解行と競合行が食い違う位置)を1にした行列を返す。
func (c *classifier) careMask(label, riv int) (*bitsx.Matrix, error) {
	if riv < 0 {
		return nil, nil
	}
	return c.q[label].Xor(c.q[riv])
}

// accumulate は分類器の更新デルタを溜める。
// 正解クラス行を現在の活性へ近づけ、最大競合クラス行を遠ざける。
func (c *classifier) accumulate(y *bitsx.Matrix, label, riv int, dl [][]int16) error {
	if !c.learnable {
		return nil
	}
	apply := func(cls, dir int) error {
		d := dl[cls]
		return y.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
			word, err := y.Word(ctx.WordIndex)
			if err != nil {
				return err
			}
			dw := d[ctx.GlobalStart:ctx.GlobalEnd]
			return ctx.ScanBits(func(i, col, colT int) error {
				// 活性の±1表現 × 方向
				pm := 2*int(word>>uint64(i)&1) - 1
				dw[i] += int16(dir * pm)
				return nil
			})
		})
	}
	if err := apply(label, 1); err != nil {
		return err
	}
	if riv >= 0 {
		return apply(riv, -1)
	}
	return nil
}

func (c *classifier) update(dl [][]int16, lr float32, rng *rand.Rand) error {
	if !c.learnable {
		return nil
	}
	for cls := range numClasses {
		q, hc, d := c.q[cls], c.h[cls], dl[cls]
		if err := q.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
			hw := hc[ctx.GlobalStart:ctx.GlobalEnd]
			dw := d[ctx.GlobalStart:ctx.GlobalEnd]
			var flips uint64
			if err := ctx.ScanBits(func(i, col, colT int) error {
				if rng.Float32() > lr {
					return nil
				}
				old := hw[i]
				clipped := int8(max(math.MinInt8, min(int(old)+int(dw[i]), math.MaxInt8)))
				hw[i] = clipped
				if (old >= 0) != (clipped >= 0) {
					flips |= 1 << uint64(i)
				}
				return nil
			}); err != nil {
				return err
			}
			old, err := q.Word(ctx.WordIndex)
			if err != nil {
				return err
			}
			return q.SetWord(ctx.WordIndex, old^flips)
		}); err != nil {
			return err
		}
	}
	return nil
}

// minPairwiseDistance は分類器行どうしの最小ハミング距離。
// 学習で行が互いに近づきすぎていないか(崩壊していないか)の診断に使う。
func (c *classifier) minPairwiseDistance() (int, float64, error) {
	minD, sum, n := math.MaxInt, 0, 0
	for i := range numClasses {
		for j := i + 1; j < numClasses; j++ {
			d, err := c.q[i].HammingDistance(c.q[j])
			if err != nil {
				return 0, 0, err
			}
			minD = min(minD, d)
			sum += d
			n++
		}
	}
	return minD, float64(sum) / float64(n), nil
}

// ---------------------------------------------------------------------------
// モデル
// ---------------------------------------------------------------------------

type model struct {
	layers []*dense
	cls    *classifier
	xRows  int
	xCols  int
}

func (m *model) outputShape() (int, int, error) {
	rows, cols := m.xRows, m.xCols
	var err error
	for i, l := range m.layers {
		rows, cols, err = l.outputShape(rows, cols)
		if err != nil {
			return 0, 0, fmt.Errorf("layer %d: %w", i, err)
		}
	}
	return rows, cols, nil
}

func (m *model) appendLayer(wRows int, rng *rand.Rand) error {
	wCols := m.xCols
	if len(m.layers) > 0 {
		_, c, err := m.outputShape()
		if err != nil {
			return err
		}
		wCols = c
	}
	l, err := newDense(wRows, wCols, rng)
	if err != nil {
		return err
	}
	m.layers = append(m.layers, l)
	return nil
}

func (m *model) forward(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) (*bitsx.Matrix, []backward, error) {
	bws := make([]backward, len(m.layers))
	var err error
	var bw backward
	for i, l := range m.layers {
		x, bw, err = l.forward(x, noiseScale, rng)
		if err != nil {
			return nil, nil, err
		}
		bws[i] = bw
	}
	return x, bws, nil
}

func (m *model) predict(x *bitsx.Matrix) (*bitsx.Matrix, error) {
	var err error
	for _, l := range m.layers {
		x, err = l.predict(x)
		if err != nil {
			return nil, err
		}
	}
	return x, nil
}

func (m *model) accuracy(xs bitsx.Matrices, labels []int, p int) (float64, error) {
	counts := make([]int, p)
	err := parallel.For(len(xs), p, func(workerID, i int) error {
		y, err := m.predict(xs[i])
		if err != nil {
			return err
		}
		lg, err := m.cls.logits(y)
		if err != nil {
			return err
		}
		best, bestVal := 0, lg[0]
		for c, v := range lg {
			if v > bestVal {
				bestVal, best = v, c
			}
		}
		if best == labels[i] {
			counts[workerID]++
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	total := 0
	for _, c := range counts {
		total += c
	}
	return float64(total) / float64(len(xs)), nil
}

// measureActivationSpread は、最終活性の級内・級間ハミング距離を測る。
//
// 注意: 精度と対応付けるため、必ず**同じエポックのモデル**で測ること。
// 学習の最後に1回だけ測ると、報告する精度(検証で選んだエポック)と
// 別のモデルの値になってしまう。
//
// 目標を P_y の完全再現にすると、同じクラスの入力はすべて同じ符号へ寄せられる
// (級内距離が小さくなる)。目標を「正解と競合の差」だけにすると、
// 指定されないビットが入力ごとにばらつくため級内距離が広がると予想される。
// プロトタイプ最近傍で読む以上、級内が広がればロジットの分散が増えて不利になる。
func measureActivationSpread(m *model, xs bitsx.Matrices, labels []int, perClass int) (float64, float64, error) {
	acts := make([][]*bitsx.Matrix, numClasses)
	for i := range xs {
		c := labels[i]
		if len(acts[c]) >= perClass {
			continue
		}
		y, err := m.predict(xs[i])
		if err != nil {
			return 0, 0, err
		}
		acts[c] = append(acts[c], y)
	}

	var withinSum, betweenSum float64
	var withinN, betweenN int
	for c := range numClasses {
		for i := range acts[c] {
			for j := i + 1; j < len(acts[c]); j++ {
				d, err := acts[c][i].HammingDistance(acts[c][j])
				if err != nil {
					return 0, 0, err
				}
				withinSum += float64(d)
				withinN++
			}
		}
		for c2 := c + 1; c2 < numClasses; c2++ {
			for i := range acts[c] {
				for j := range acts[c2] {
					d, err := acts[c][i].HammingDistance(acts[c2][j])
					if err != nil {
						return 0, 0, err
					}
					betweenSum += float64(d)
					betweenN++
				}
			}
		}
	}
	return withinSum / float64(withinN), betweenSum / float64(betweenN), nil
}

// ---------------------------------------------------------------------------
// 学習
// ---------------------------------------------------------------------------

type trainer struct {
	model         *model
	miniBatchSize int
	lr            float32
	clsLR         float32
	margin        float32
	noiseScale    float32
	targetMode    string
	selectN       int  // 目標を指定するビット数(<=0 で母集団すべて)
	hardMask      bool // 選ばれなかったビットをゲートからも外すか

	workerRNGs []*rand.Rand
	shuffleRNG *rand.Rand
	updateRNG  *rand.Rand

	workerDeltas [][]*delta
	aggregated   []*delta
	workerClsDs  [][][]int16
	aggClsD      [][]int16
}

func newTrainer(m *model, p int, seed uint64) (*trainer, error) {
	if p <= 0 {
		return nil, errors.New("ワーカー数は1以上であるべき")
	}
	rngs := make([]*rand.Rand, p)
	for i := range rngs {
		rngs[i] = rand.New(rand.NewPCG(seed, 0x9E3779B97F4A7C15+uint64(i)))
	}
	wd := make([][]*delta, p)
	wc := make([][][]int16, p)
	n := m.cls.totalBits()
	for i := range p {
		ds := make([]*delta, len(m.layers))
		for l, layer := range m.layers {
			ds[l] = layer.newDelta()
		}
		wd[i] = ds
		cd := make([][]int16, numClasses)
		for c := range numClasses {
			cd[c] = make([]int16, n)
		}
		wc[i] = cd
	}
	agg := make([]*delta, len(m.layers))
	for l, layer := range m.layers {
		agg[l] = layer.newDelta()
	}
	aggCls := make([][]int16, numClasses)
	for c := range numClasses {
		aggCls[c] = make([]int16, n)
	}
	return &trainer{
		model: m, miniBatchSize: 1024, lr: 0.1, clsLR: 0.1, margin: 0.5,
		noiseScale: 0.5, targetMode: "proto", selectN: 0, hardMask: false,
		workerRNGs:   rngs,
		shuffleRNG:   rand.New(rand.NewPCG(seed, 0xD1B54A32D192ED03)),
		updateRNG:    rand.New(rand.NewPCG(seed, 0xA24BAED4963EE407)),
		workerDeltas: wd, aggregated: agg, workerClsDs: wc, aggClsD: aggCls,
	}, nil
}

// satisfiesUpdateCriterion は ../train.go と同じ判定を、
// 既に求めたロジットに対して行う版。
func satisfiesUpdateCriterion(logits []int, label, riv int, totalBits int, margin float32) bool {
	if riv < 0 {
		return false
	}
	// 一致数の差。../train.go の marginBits と同じスケール
	marginBits := int(float32(totalBits) * margin / 2)
	return logits[label]-logits[riv] < marginBits
}

func (t *trainer) trainEpoch(xs bitsx.Matrices, labels []int) error {
	n := len(xs)
	batch := min(t.miniBatchSize, n)
	perm := t.shuffleRNG.Perm(n)
	totalBits := t.model.cls.totalBits()

	for start := 0; start < n; start += batch {
		end := min(start+batch, n)
		idxs := perm[start:end]

		for _, ds := range t.workerDeltas {
			for _, d := range ds {
				d.clear()
			}
		}
		for _, cd := range t.workerClsDs {
			for _, d := range cd {
				clear(d)
			}
		}

		p := len(t.workerRNGs)
		if err := parallel.For(len(idxs), p, func(workerID, i int) error {
			rng := t.workerRNGs[workerID]
			x := xs[idxs[i]]
			label := labels[idxs[i]]

			y, bws, err := t.model.forward(x, t.noiseScale, rng)
			if err != nil {
				return err
			}
			lg, err := t.model.cls.logits(y)
			if err != nil {
				return err
			}
			riv := rival(lg, label)
			if !satisfiesUpdateCriterion(lg, label, riv, totalBits, t.margin) {
				return nil
			}

			if err := t.model.cls.accumulate(y, label, riv, t.workerClsDs[workerID]); err != nil {
				return err
			}
			sel, err := t.model.cls.selectionMask(label, riv, t.targetMode, t.selectN, rng)
			if err != nil {
				return err
			}
			target, err := t.model.cls.targetFrom(y, label, sel)
			if err != nil {
				return err
			}
			// hard なら、選ばれなかったビットを後方射影(ゲート)からも外す
			var care *bitsx.Matrix
			if t.hardMask {
				care = sel
			}
			last := len(bws) - 1
			for li := range slices.Backward(bws) {
				c := care
				if li != last {
					c = nil // careは最終層にだけ効く
				}
				target, err = bws[li](target, c, t.workerDeltas[workerID][li])
				if err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return err
		}

		for _, d := range t.aggregated {
			d.clear()
		}
		for _, d := range t.aggClsD {
			clear(d)
		}
		for _, ds := range t.workerDeltas {
			for li, d := range ds {
				t.aggregated[li].add(d)
			}
		}
		for _, cd := range t.workerClsDs {
			for c, d := range cd {
				for i, v := range d {
					t.aggClsD[c][i] += v
				}
			}
		}
		for _, d := range t.aggregated {
			d.sign()
		}
		for _, d := range t.aggClsD {
			for i, v := range d {
				d[i] = int16(cmp.Compare(v, 0))
			}
		}

		for li, layer := range t.model.layers {
			if err := layer.update(t.aggregated[li], t.lr, t.updateRNG); err != nil {
				return err
			}
		}
		if err := t.model.cls.update(t.aggClsD, t.clsLR, t.updateRNG); err != nil {
			return err
		}
	}
	return nil
}

// ---------------------------------------------------------------------------

func splitTrainValidation(xs bitsx.Matrices, labels []int, valRatio float64, rng *rand.Rand) (
	bitsx.Matrices, []int, bitsx.Matrices, []int, error) {
	n := len(xs)
	valN := int(float64(n) * valRatio)
	if valN <= 0 || valN >= n {
		return nil, nil, nil, nil, fmt.Errorf("valRatio %g では分割できません (n = %d)", valRatio, n)
	}
	perm := rng.Perm(n)
	valXs := make(bitsx.Matrices, 0, valN)
	valLabels := make([]int, 0, valN)
	trXs := make(bitsx.Matrices, 0, n-valN)
	trLabels := make([]int, 0, n-valN)
	for i, idx := range perm {
		if i < valN {
			valXs = append(valXs, xs[idx])
			valLabels = append(valLabels, labels[idx])
		} else {
			trXs = append(trXs, xs[idx])
			trLabels = append(trLabels, labels[idx])
		}
	}
	return trXs, trLabels, valXs, valLabels, nil
}

func main() {
	var (
		dsName     = flag.String("dataset", "mnist", "mnist | fashion")
		targetMode = flag.String("target", "proto", "proto | diff | randall | randdiff")
		selectN    = flag.Int("n", 0, "目標を指定するビット数(<=0 で母集団すべて)")
		selectQ    = flag.Float64("q", 0, "nの代わりに母集団に対する割合で指定(>0で優先)")
		hardMask   = flag.Bool("hard", false, "選ばれなかったビットをゲートからも外す")
		learnQ     = flag.Bool("learnq", false, "出力分類器を学習する")
		h1         = flag.Int("h1", 512, "隠れ層1の幅(0で省略)")
		h2         = flag.Int("h2", 1024, "隠れ層2(最終)の幅")
		lr         = flag.Float64("lr", 0.1, "バックボーンの確率的学習率")
		clsLR      = flag.Float64("clslr", 0.1, "分類器の確率的学習率")
		margin     = flag.Float64("margin", 0.5, "更新判定のマージン(論文のrスケール)")
		groupSize  = flag.Int("gsize", 4, "GroupSize")
		gateScale  = flag.Float64("gate", 1.0, "GateDropThresholdScale")
		noiseScale = flag.Float64("noise", 0.5, "NoiseStdScale")
		epochs     = flag.Int("epochs", 20, "エポック数")
		batch      = flag.Int("batch", 1024, "ミニバッチサイズ")
		valRatio   = flag.Float64("valratio", 0.1, "学習データから検証用に分ける割合")
		seed       = flag.Uint64("seed", 1, "乱数シード")
		threads    = flag.Int("threads", 0, "ワーカー数 (0 = NumCPU)")
	)
	flag.Parse()

	workers := *threads
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	switch *targetMode {
	case "proto", "diff", "randall", "randdiff":
	default:
		log.Fatalf("targetが不正: %s", *targetMode)
	}

	var ds dataset.Binary[int]
	var err error
	if *dsName == "fashion" {
		ds, err = dataset.LoadFashionMNIST(nil)
	} else {
		ds, err = dataset.LoadMNIST(nil)
	}
	if err != nil {
		log.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(*seed, *seed+1))
	m := &model{xRows: 1, xCols: 784}
	if *h1 > 0 {
		if err := m.appendLayer(*h1, rng); err != nil {
			log.Fatal(err)
		}
	}
	if err := m.appendLayer(*h2, rng); err != nil {
		log.Fatal(err)
	}
	yRows, yCols, err := m.outputShape()
	if err != nil {
		log.Fatal(err)
	}
	totalBits := numClasses * yRows * yCols
	iters := 10 * int(float64(totalBits)*math.Log(float64(totalBits)))
	protos, err := bitsx.NewETFMatrices(numClasses, yRows, yCols, iters, rng)
	if err != nil {
		log.Fatal(err)
	}
	cls, err := newClassifier(protos, *learnQ)
	if err != nil {
		log.Fatal(err)
	}
	m.cls = cls
	for _, l := range m.layers {
		l.groupSize = *groupSize
		l.gateScale = float32(*gateScale)
	}

	tr, err := newTrainer(m, workers, *seed)
	if err != nil {
		log.Fatal(err)
	}
	tr.miniBatchSize = *batch
	tr.lr = float32(*lr)
	tr.clsLR = float32(*clsLR)
	tr.margin = float32(*margin)
	tr.noiseScale = float32(*noiseScale)
	tr.targetMode = *targetMode
	tr.hardMask = *hardMask
	tr.selectN = *selectN
	if *selectQ > 0 {
		// 母集団の大きさは diff 系なら概ね総ビットの半分、all 系なら総ビット
		pool := cls.totalBits()
		if *targetMode == "randdiff" || *targetMode == "diff" {
			pool /= 2
		}
		tr.selectN = max(1, int(*selectQ*float64(pool)))
	}

	splitRNG := rand.New(rand.NewPCG(*seed, 0xC2B2AE3D27D4EB4F))
	trainXs, trainLabels, valXs, valLabels, err := splitTrainValidation(
		ds.TrainInputs, ds.TrainLabels, *valRatio, splitRNG)
	if err != nil {
		log.Fatal(err)
	}

	minD0, meanD0, err := cls.minPairwiseDistance()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("dataset=%s target=%s n=%d hard=%t learnq=%t 構成=784->%d->%d lr=%g clslr=%g margin=%g gsize=%d gate=%g noise=%g epochs=%d seed=%d train=%d val=%d test=%d\n",
		*dsName, *targetMode, tr.selectN, *hardMask, *learnQ, *h1, *h2, *lr, *clsLR, *margin, *groupSize,
		*gateScale, *noiseScale, *epochs, *seed, len(trainXs), len(valXs), len(ds.TestInputs))
	fmt.Printf("分類器行の距離(初期): 最小 %d / 平均 %.1f / 総ビット %d\n", minD0, meanD0, cls.totalBits())

	bestVal, testAtBestVal, bestEpoch := -1.0, 0.0, 0
	bestWithin, bestBetween := 0.0, 0.0
	for e := 1; e <= *epochs; e++ {
		t0 := time.Now()
		if err := tr.trainEpoch(trainXs, trainLabels); err != nil {
			log.Fatal(err)
		}
		valAcc, err := m.accuracy(valXs, valLabels, workers)
		if err != nil {
			log.Fatal(err)
		}
		testAcc, err := m.accuracy(ds.TestInputs, ds.TestLabels, workers)
		if err != nil {
			log.Fatal(err)
		}
		// 級内/級間は精度と同じモデルで測る必要があるため、毎エポック測って
		// 検証が最良だったエポックの値を保持する
		within, between, err := measureActivationSpread(m, ds.TestInputs, ds.TestLabels, 60)
		if err != nil {
			log.Fatal(err)
		}
		if valAcc > bestVal {
			bestVal, testAtBestVal, bestEpoch = valAcc, testAcc, e
			bestWithin, bestBetween = within, between
		}
		fmt.Printf("epoch %d: val acc %.4f / test acc %.4f (best val %.4f @%d) %.1fs\n",
			e, valAcc, testAcc, bestVal, bestEpoch, time.Since(t0).Seconds())
	}

	minD, meanD, err := cls.minPairwiseDistance()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("分類器行の距離(学習後): 最小 %d / 平均 %.1f\n", minD, meanD)
	fmt.Printf("最終活性の距離(検証最良エポックのモデル): 級内 %.1f / 級間 %.1f / 比 %.3f (総ビット %d)\n",
		bestWithin, bestBetween, bestWithin/bestBetween, cls.totalBits())
	fmt.Printf("BEST_VAL %.4f @epoch %d / TEST %.4f\n", bestVal, bestEpoch, testAtBestVal)
}
