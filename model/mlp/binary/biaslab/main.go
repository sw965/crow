// biaslab: BEP に「学習する整数バイアス」を追加したときの効果を測る実験プログラム。
//
// 依頼された学習則:
//   - 各層のニューロンに整数バイアスを持たせる
//   - Update のたびに、ニューロンごとに「バイアスを動かす」か「重みを動かす」かを
//     乱数で選び、**両方は動かさない**
//   - バイアスが選ばれたら ±1 動かす
//   - 重みが選ばれたら crow の従来どおり(確率 LR で隠れ重み H に集約デルタを加算し、
//     H の符号が変わったら可視重み W のビットを反転)
//
// これに加えて、見本(プロトタイプ)へのセルマスクの有無も比較する。
// セルマスクは ../PROTOTYPES.md B-9 の手法で、全プロトタイプに共通の
// 無作為なセル単位マスクを XOR するもの。列単位のマスクと違い、
// ニューロンの重み反転では元に戻せないため対称変換ではない。
//
// 分類はこれまでどおり argmax、回帰は加重平均のみで復号する(../PROTOTYPES.md B-10b)。
//
// crow / omw のライブラリコードは変更していない。BEP の順伝播・逆伝播・更新は
// ../layer.go ../train.go と同じ手順を bitsx の公開APIで再実装したもので、
// バイアスを無効(-bias=false)にすると crow の BEP と同じ計算になる。
//
// 実行例:
//
//	go run . -task classify -dataset mnist   -bias -cellmask
//	go run . -task classify -dataset fashion -bias=false -cellmask=false
//	go run . -task regress   -bias -cellmask
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

const hInitAbs = 4 // ../layer.go と同じ

// ---------------------------------------------------------------------------
// 層
// ---------------------------------------------------------------------------

type delta struct {
	w    []int16 // wRows * wCols
	bias []int16 // wRows
	// 選ばれた更新対象の希望活性の極性。バイアスが一方向へ偏る原因の診断用。
	posDesired int64
	negDesired int64
}

func newDelta(wRows, wCols int) *delta {
	return &delta{w: make([]int16, wRows*wCols), bias: make([]int16, wRows)}
}

func (d *delta) clear() {
	clear(d.w)
	clear(d.bias)
	d.posDesired = 0
	d.negDesired = 0
}

func (d *delta) add(other *delta) {
	for i, v := range other.w {
		d.w[i] += v
	}
	for i, v := range other.bias {
		d.bias[i] += v
	}
	d.posDesired += other.posDesired
	d.negDesired += other.negDesired
}

func (d *delta) sign() {
	for i, v := range d.w {
		d.w[i] = int16(cmp.Compare(v, 0))
	}
	for i, v := range d.bias {
		d.bias[i] = int16(cmp.Compare(v, 0))
	}
}

// dense は ../layer.go の Dense に整数バイアスを足したもの。
type dense struct {
	w  *bitsx.Matrix
	wt *bitsx.Matrix
	h  []int8
	// bias はニューロンごとの整数しきい値。z に加算される。
	bias    []int32
	biasMax int32

	gateBase int
	noiseStd float32

	gateScale  float32
	groupSize  int
	useBias    bool
	biasChoice float32 // Update時にバイアス側が選ばれる確率
}

func newDense(wRows, wCols int, useBias bool, biasChoice float32, rng *rand.Rand) (*dense, error) {
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
		bias:       make([]int32, wRows),
		biasMax:    int32(wCols), // |bias| + fanIn がアキュムレータ幅を超えない範囲
		gateBase:   int(noiseStdBase),
		noiseStd:   noiseStdBase,
		gateScale:  1.0,
		groupSize:  4,
		useBias:    useBias,
		biasChoice: biasChoice,
	}, nil
}

func (d *dense) outputShape(xRows, xCols int) (int, int, error) {
	if xCols != d.w.Cols() {
		return 0, 0, fmt.Errorf("入力の列数が不一致: xCols = %d, W.Cols = %d", xCols, d.w.Cols())
	}
	return xRows, d.w.Rows(), nil
}

func (d *dense) newDelta() *delta { return newDelta(d.w.Rows(), d.w.Cols()) }

// preActivation は z を求める。noiseStd > 0 なら学習時のノイズを加える。
func (d *dense) preActivation(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) ([]int, int, int, error) {
	u, err := x.Dot(d.w)
	if err != nil {
		return nil, 0, 0, err
	}
	fanIn := d.w.Cols()
	yCols := d.w.Rows()
	z := make([]int, len(u))

	std := noiseScale * d.noiseStd
	for i, count := range u {
		zi := 2*count - fanIn
		if d.useBias {
			zi += int(d.bias[i%yCols])
		}
		if std > 0 {
			noise, err := randx.IntNorm(-fanIn, fanIn, 0, std, rng)
			if err != nil {
				return nil, 0, 0, err
			}
			zi += noise
		}
		z[i] = zi
	}
	return z, x.Rows(), yCols, nil
}

type backward func(t *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error)

func (d *dense) forward(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) (*bitsx.Matrix, backward, error) {
	z, yRows, yCols, err := d.preActivation(x, noiseScale, rng)
	if err != nil {
		return nil, nil, err
	}
	y, err := bitsx.NewSignMatrix(yRows, yCols, z)
	if err != nil {
		return nil, nil, err
	}

	bw := func(t *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error) {
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

			if err := keepGate.SetWord(tCtx.WordIndex, keepGateWord); err != nil {
				return err
			}

			slices.SortFunc(mismatches, func(a, b wordMismatch) int {
				return cmp.Compare(a.absZi, b.absZi)
			})

			updateK := max(len(zWord)/d.groupSize, 1)
			updateK = min(updateK, len(mismatches))

			for _, mm := range mismatches[:updateK] {
				// 希望活性の±1表現。バイアスのデルタはこれをそのまま使う
				// (../UNIVERSAL_APPROXIMATION.md §5.2 の sampleDeltaBias = a*)。
				if mm.tBit == 1 {
					dl.bias[mm.col]++
					dl.posDesired++
				} else {
					dl.bias[mm.col]--
					dl.negDesired++
				}

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

// update は、ニューロンごとに「バイアスを動かす」か「重みを動かす」かを乱数で選び、
// 選ばれた方だけを更新する。
//
// 重要: 選択は useBias の値によらず同じ確率で行う。
// useBias=false のときは「バイアス側が選ばれたニューロンは何も更新しない」となり、
// バイアス有りと**重みの更新機会が揃った対照条件**になる。
// biasChoice=0 なら従来のBEP(全ニューロンで重みを更新)と同じ。
func (d *dense) update(dl *delta, lr float32, rng *rand.Rand) error {
	wRows := d.w.Rows()

	// ニューロン単位で更新対象を決める(1行が複数ワードにまたがるため、事前に決めておく)。
	// biasChoice <= 0 のときは乱数を1つも消費しない。これにより crow の Update と
	// 乱数消費が完全に一致し、差分テストで1ステップ比較できる。
	biasMode := make([]bool, wRows)
	if d.biasChoice > 0 {
		for i := range biasMode {
			biasMode[i] = rng.Float32() < d.biasChoice
		}
	}

	// バイアス側(useBias=false なら何もしない = 重み更新を落とすだけの対照)
	if d.useBias {
		for i := range wRows {
			if !biasMode[i] {
				continue
			}
			// 集約済みデルタは Sign 済みなので ±1 か 0
			d.bias[i] = max(-d.biasMax, min(d.bias[i]+int32(dl.bias[i]), d.biasMax))
		}
	}

	// 重み側(crow の従来どおり)
	return d.w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		if biasMode[ctx.Row] {
			return nil
		}
		hWord := d.h[ctx.GlobalStart:ctx.GlobalEnd]
		dWord := dl.w[ctx.GlobalStart:ctx.GlobalEnd]
		var flips uint64
		if err := ctx.ScanBits(func(i, col, colT int) error {
			if rng.Float32() > lr {
				return nil
			}
			old := hWord[i]
			newVal := int(old) + int(dWord[i])
			clipped := int8(max(math.MinInt8, min(newVal, math.MaxInt8)))
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
// モデル
// ---------------------------------------------------------------------------

type model struct {
	layers     []*dense
	prototypes bitsx.Matrices
	values     []float64
	xRows      int
	xCols      int
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

func (m *model) appendLayer(wRows int, useBias bool, biasChoice float32, rng *rand.Rand) error {
	wCols := m.xCols
	if len(m.layers) > 0 {
		_, c, err := m.outputShape()
		if err != nil {
			return err
		}
		wCols = c
	}
	l, err := newDense(wRows, wCols, useBias, biasChoice, rng)
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

func (m *model) logits(x *bitsx.Matrix) ([]int, error) {
	y, err := m.predict(x)
	if err != nil {
		return nil, err
	}
	total := y.Rows() * y.Cols()
	out := make([]int, len(m.prototypes))
	for i, p := range m.prototypes {
		hd, err := y.HammingDistance(p)
		if err != nil {
			return nil, err
		}
		out[i] = total - hd
	}
	return out, nil
}

// applyCellMask は、全プロトタイプに共通のセル単位マスクを XOR する(../PROTOTYPES.md B-9)。
func applyCellMask(protos bitsx.Matrices, rng *rand.Rand) error {
	if len(protos) == 0 {
		return errors.New("prototypesが空です")
	}
	mask, err := bitsx.NewRandMatrix(protos[0].Rows(), protos[0].Cols(), 0, rng)
	if err != nil {
		return err
	}
	for i, p := range protos {
		masked, err := p.Xor(mask)
		if err != nil {
			return err
		}
		protos[i] = masked
	}
	return nil
}

// ---------------------------------------------------------------------------
// 学習
// ---------------------------------------------------------------------------

type trainer struct {
	model         *model
	miniBatchSize int
	lr            float32
	margin        float32
	noiseScale    float32
	workerRNGs    []*rand.Rand
	shuffleRNG    *rand.Rand
	// updateRNG は Update の乱数。層をまたいで1本にすることで、
	// 層数やワーカー数に依存せず seed だけで再現できるようにする。
	updateRNG    *rand.Rand
	workerDeltas [][]*delta
	aggregated   []*delta

	// 希望活性の極性の累計(診断用)
	posDesired int64
	negDesired int64
}

// newTrainer は学習器を作る。学習に使う乱数(シャッフル・ノイズ・更新選択)は
// すべて seed から決定的に導出する。randx.NewPCG はグローバル乱数から
// シードするため実行ごとに結果が変わってしまうので、ここでは使わない
// (../PROTOTYPES.md B-8b が crow 本体について指摘しているのと同じ問題)。
func newTrainer(m *model, p int, seed uint64) (*trainer, error) {
	if p <= 0 {
		return nil, errors.New("ワーカー数は1以上であるべき")
	}
	rngs := make([]*rand.Rand, p)
	for i := range rngs {
		rngs[i] = rand.New(rand.NewPCG(seed, 0x9E3779B97F4A7C15+uint64(i)))
	}
	wd := make([][]*delta, p)
	for i := range wd {
		ds := make([]*delta, len(m.layers))
		for l, layer := range m.layers {
			ds[l] = layer.newDelta()
		}
		wd[i] = ds
	}
	agg := make([]*delta, len(m.layers))
	for l, layer := range m.layers {
		agg[l] = layer.newDelta()
	}
	return &trainer{
		model: m, miniBatchSize: 128, lr: 0.1, margin: 0.5, noiseScale: 0.5,
		workerRNGs:   rngs,
		shuffleRNG:   rand.New(rand.NewPCG(seed, 0xD1B54A32D192ED03)),
		updateRNG:    rand.New(rand.NewPCG(seed, 0xA24BAED4963EE407)),
		workerDeltas: wd, aggregated: agg,
	}, nil
}

// satisfiesUpdateCriterion は ../train.go と同じ判定。
func satisfiesUpdateCriterion(y *bitsx.Matrix, label int, protos bitsx.Matrices, margin float32) (bool, error) {
	t := protos[label]
	yMismatch, err := y.HammingDistance(t)
	if err != nil {
		return false, err
	}
	totalBits := y.Rows() * y.Cols()
	marginBits := int(float32(totalBits) * margin / 2)
	for i, p := range protos {
		if i == label {
			continue
		}
		mismatch, err := y.HammingDistance(p)
		if err != nil {
			return false, err
		}
		if mismatch-yMismatch < marginBits {
			return true, nil
		}
	}
	return false, nil
}

func (t *trainer) trainEpoch(xs bitsx.Matrices, labels []int) error {
	n := len(xs)
	batch := min(t.miniBatchSize, n)
	perm := t.shuffleRNG.Perm(n)

	for start := 0; start < n; start += batch {
		end := min(start+batch, n)
		idxs := perm[start:end]

		for _, ds := range t.workerDeltas {
			for _, d := range ds {
				d.clear()
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
			should, err := satisfiesUpdateCriterion(y, label, t.model.prototypes, t.margin)
			if err != nil {
				return err
			}
			if !should {
				return nil
			}
			target := t.model.prototypes[label]
			for li := range slices.Backward(bws) {
				target, err = bws[li](target, t.workerDeltas[workerID][li])
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
		for _, ds := range t.workerDeltas {
			for li, d := range ds {
				t.aggregated[li].add(d)
			}
		}
		for _, d := range t.aggregated {
			t.posDesired += d.posDesired
			t.negDesired += d.negDesired
			d.sign()
		}

		for li, layer := range t.model.layers {
			if err := layer.update(t.aggregated[li], t.lr, t.updateRNG); err != nil {
				return err
			}
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// 評価
// ---------------------------------------------------------------------------

func (m *model) accuracy(xs bitsx.Matrices, labels []int, p int) (float64, error) {
	counts := make([]int, p)
	err := parallel.For(len(xs), p, func(workerID, i int) error {
		lg, err := m.logits(xs[i])
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

// weightedAverage は加重平均による復号(../PROTOTYPES.md B-10b)。
// 温度 T でロジットを重みに変換し、値の加重平均を返す。
// 復号は評価のための処理であり、学習カーネルの外側にある(浮動小数点を使う)。
func weightedAverage(logits []int, values []float64, temperature float64) float64 {
	maxLogit := slices.Max(logits)
	var num, den float64
	for i, l := range logits {
		w := math.Exp(float64(l-maxLogit) / temperature)
		num += w * values[i]
		den += w
	}
	return num / den
}

// allLogits は全サンプルのロジットをまとめて求める。
// 温度の掃引でロジットを何度も計算し直さないためのキャッシュ。
func (m *model) allLogits(xs bitsx.Matrices, p int) ([][]int, error) {
	out := make([][]int, len(xs))
	err := parallel.For(len(xs), p, func(workerID, i int) error {
		lg, err := m.logits(xs[i])
		if err != nil {
			return err
		}
		out[i] = lg
		return nil
	})
	return out, err
}

func (m *model) maeFromLogits(logits [][]int, levels []int, temperature float64) float64 {
	var total float64
	for i, lg := range logits {
		pred := weightedAverage(lg, m.values, temperature)
		total += math.Abs(pred - m.values[levels[i]])
	}
	return total / float64(len(logits))
}

// bestTemperature は、学習データだけを使って加重平均の温度を選ぶ(../PROTOTYPES.md B-10a)。
func (m *model) bestTemperature(logits [][]int, levels []int) (float64, float64) {
	candidates := []float64{5, 10, 20, 40, 60, 80, 120, 160, 240, 320, 480}
	bestT, bestMAE := candidates[0], math.Inf(1)
	for _, tmp := range candidates {
		v := m.maeFromLogits(logits, levels, tmp)
		if v < bestMAE {
			bestMAE, bestT = v, tmp
		}
	}
	return bestT, bestMAE
}

// splitTrainValidation は、xs を学習用と検証用へ分ける。
// 最良エポックとハイパーパラメータの選択を検証集合で行い、
// テスト集合は最後に1回だけ見るための下準備。
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

// reportPreActivationStd は、活性前値の標準偏差を実測する。
// √fanIn は「重みと入力が独立ランダム」という仮定の下での値でしかないので、
// バイアスの大きさを比較する基準には実測値を使う。
func reportPreActivationStd(m *model, xs bitsx.Matrices, workers int) error {
	sampleN := min(len(xs), 512)
	for li, l := range m.layers {
		var sum, sumSq float64
		var count int
		for i := range sampleN {
			// この層までの入力を作る
			x := xs[i]
			for _, prev := range m.layers[:li] {
				var err error
				x, err = prev.predict(x)
				if err != nil {
					return err
				}
			}
			u, err := x.Dot(l.w)
			if err != nil {
				return err
			}
			fanIn := l.w.Cols()
			for _, c := range u {
				// バイアス抜きの活性前値(バイアスと比較する基準)
				z := float64(2*c - fanIn)
				sum += z
				sumSq += z * z
				count++
			}
		}
		mean := sum / float64(count)
		std := math.Sqrt(max(sumSq/float64(count)-mean*mean, 0))
		fmt.Printf("  層%d 活性前値(バイアス抜き): 平均 %.2f / 標準偏差(実測) %.2f / √fanIn %.2f\n",
			li, mean, std, math.Sqrt(float64(l.w.Cols())))
	}
	return nil
}

// reportBias は、学習後のバイアスがどれだけ動いたかを出力する。
// 「バイアスが効かなかった」のか「そもそも動いていない」のかを切り分けるための診断。
func reportBias(m *model) {
	for li, l := range m.layers {
		if !l.useBias {
			continue
		}
		var sum, absSum float64
		maxAbs := int32(0)
		nonZero := 0
		for _, b := range l.bias {
			sum += float64(b)
			ab := b
			if ab < 0 {
				ab = -ab
			}
			absSum += float64(ab)
			maxAbs = max(maxAbs, ab)
			if b != 0 {
				nonZero++
			}
		}
		n := float64(len(l.bias))
		fmt.Printf("  層%d バイアス: 平均 %.2f / 平均絶対値 %.2f / 最大絶対値 %d / 非ゼロ %d/%d / fanIn %d\n",
			li, sum/n, absSum/n, maxAbs, nonZero, len(l.bias), l.w.Cols())
	}
}

// ---------------------------------------------------------------------------
// 回帰用の合成タスク(../PROTOTYPES.md B-8 と同じ生成規則)
// ---------------------------------------------------------------------------

func newRegressionData(n, count, rows, cols int, d *bitsx.Matrix, rng *rand.Rand) (bitsx.Matrices, []int, error) {
	xs := make(bitsx.Matrices, count)
	levels := make([]int, count)
	for i := range count {
		level := i % n
		levels[i] = level
		// q(l) = 0.35 + 0.30 * l/(n-1) を整数比較で判定する
		qNum := 350000 + 300000*level/(n-1)
		x, err := bitsx.NewZerosMatrix(rows, cols)
		if err != nil {
			return nil, nil, err
		}
		if err := x.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
			dWord, err := d.Word(ctx.WordIndex)
			if err != nil {
				return err
			}
			var word uint64
			if err := ctx.ScanBits(func(bi, col, colT int) error {
				bit := dWord >> uint(bi) & 1
				if rng.IntN(1000000) >= qNum {
					bit ^= 1 // 確率 1-q で反転
				}
				if bit == 1 {
					word |= 1 << uint(bi)
				}
				return nil
			}); err != nil {
				return err
			}
			return x.SetWord(ctx.WordIndex, word)
		}); err != nil {
			return nil, nil, err
		}
		xs[i] = x
	}
	return xs, levels, nil
}

// ---------------------------------------------------------------------------

type config struct {
	useBias    bool
	cellMask   bool
	biasChoice float32
	lr         float32
	margin     float32
	groupSize  int
	gateScale  float32
	noiseScale float32
	epochs     int
	batch      int
	valRatio   float64
	seed       uint64
}

func runClassification(dsName string, cfg config, workers int) error {
	var ds dataset.Binary[int]
	var err error
	if dsName == "fashion" {
		ds, err = dataset.LoadFashionMNIST(nil)
	} else {
		ds, err = dataset.LoadMNIST(nil)
	}
	if err != nil {
		return err
	}

	rng := rand.New(rand.NewPCG(cfg.seed, cfg.seed+1))
	m := &model{xRows: 1, xCols: 784}
	if err := m.appendLayer(512, cfg.useBias, cfg.biasChoice, rng); err != nil {
		return err
	}
	if err := m.appendLayer(1024, cfg.useBias, cfg.biasChoice, rng); err != nil {
		return err
	}
	yRows, yCols, err := m.outputShape()
	if err != nil {
		return err
	}
	totalBits := 10 * yRows * yCols
	iters := 10 * int(float64(totalBits)*math.Log(float64(totalBits)))
	protos, err := bitsx.NewETFMatrices(10, yRows, yCols, iters, rng)
	if err != nil {
		return err
	}
	m.prototypes = protos
	if cfg.cellMask {
		if err := applyCellMask(m.prototypes, rng); err != nil {
			return err
		}
	}
	for _, l := range m.layers {
		l.groupSize = cfg.groupSize
		l.gateScale = cfg.gateScale
	}

	tr, err := newTrainer(m, workers, cfg.seed)
	if err != nil {
		return err
	}
	tr.miniBatchSize = cfg.batch
	tr.lr = cfg.lr
	tr.margin = cfg.margin
	tr.noiseScale = cfg.noiseScale

	// 最良エポックは検証集合で選ぶ。テスト集合は選択に使わない。
	splitRNG := rand.New(rand.NewPCG(cfg.seed, 0xC2B2AE3D27D4EB4F))
	trainXs, trainLabels, valXs, valLabels, err := splitTrainValidation(
		ds.TrainInputs, ds.TrainLabels, cfg.valRatio, splitRNG)
	if err != nil {
		return err
	}

	fmt.Printf("task=classify dataset=%s bias=%t cellmask=%t biaschoice=%g lr=%g margin=%g gsize=%d gate=%g noise=%g epochs=%d valratio=%g seed=%d train=%d val=%d test=%d\n",
		dsName, cfg.useBias, cfg.cellMask, cfg.biasChoice, cfg.lr, cfg.margin,
		cfg.groupSize, cfg.gateScale, cfg.noiseScale, cfg.epochs, cfg.valRatio, cfg.seed,
		len(trainXs), len(valXs), len(ds.TestInputs))

	bestVal, testAtBestVal, bestEpoch := -1.0, 0.0, 0
	for e := 1; e <= cfg.epochs; e++ {
		t0 := time.Now()
		if err := tr.trainEpoch(trainXs, trainLabels); err != nil {
			return err
		}
		valAcc, err := m.accuracy(valXs, valLabels, workers)
		if err != nil {
			return err
		}
		testAcc, err := m.accuracy(ds.TestInputs, ds.TestLabels, workers)
		if err != nil {
			return err
		}
		if valAcc > bestVal {
			bestVal, testAtBestVal, bestEpoch = valAcc, testAcc, e
		}
		fmt.Printf("epoch %d: val acc %.4f / test acc %.4f (best val %.4f @%d) %.1fs\n",
			e, valAcc, testAcc, bestVal, bestEpoch, time.Since(t0).Seconds())
	}
	if err := reportPreActivationStd(m, ds.TestInputs, workers); err != nil {
		return err
	}
	reportBias(m)
	fmt.Printf("希望活性の極性(累計): +1 = %d / -1 = %d\n", tr.posDesired, tr.negDesired)
	fmt.Printf("BEST_VAL %.4f @epoch %d / TEST %.4f\n", bestVal, bestEpoch, testAtBestVal)
	return nil
}

func runRegression(cfg config, workers int) error {
	const (
		levels  = 101
		xRows   = 64
		xCols   = 128
		trainN  = 4040 // levels の倍数
		testN   = 1010
		hidden  = 256
		outCols = 64
	)

	rng := rand.New(rand.NewPCG(cfg.seed, cfg.seed+1))
	d, err := bitsx.NewRandMatrix(xRows, xCols, 0, rng)
	if err != nil {
		return err
	}
	trainXs, trainLevels, err := newRegressionData(levels, trainN, xRows, xCols, d, rng)
	if err != nil {
		return err
	}
	testXs, testLevels, err := newRegressionData(levels, testN, xRows, xCols, d, rng)
	if err != nil {
		return err
	}

	m := &model{xRows: xRows, xCols: xCols}
	if err := m.appendLayer(hidden, cfg.useBias, cfg.biasChoice, rng); err != nil {
		return err
	}
	if err := m.appendLayer(outCols, cfg.useBias, cfg.biasChoice, rng); err != nil {
		return err
	}
	yRows, yCols, err := m.outputShape()
	if err != nil {
		return err
	}
	protos, err := bitsx.NewThermometerMatrices(levels, yRows, yCols)
	if err != nil {
		return err
	}
	m.prototypes = protos
	if cfg.cellMask {
		if err := applyCellMask(m.prototypes, rng); err != nil {
			return err
		}
	}
	m.values = make([]float64, levels)
	for i := range levels {
		m.values[i] = float64(i)
	}
	for _, l := range m.layers {
		l.groupSize = cfg.groupSize
		l.gateScale = cfg.gateScale
	}

	tr, err := newTrainer(m, workers, cfg.seed)
	if err != nil {
		return err
	}
	tr.miniBatchSize = cfg.batch
	tr.lr = cfg.lr
	tr.margin = cfg.margin
	tr.noiseScale = cfg.noiseScale

	// 温度と最良エポックは検証集合で選ぶ。テスト集合は選択に使わない。
	splitRNG := rand.New(rand.NewPCG(cfg.seed, 0xC2B2AE3D27D4EB4F))
	trXs, trLevels, valXs, valLevels, err := splitTrainValidation(trainXs, trainLevels, cfg.valRatio, splitRNG)
	if err != nil {
		return err
	}

	fmt.Printf("task=regress levels=%d bias=%t cellmask=%t biaschoice=%g lr=%g margin=%g gsize=%d gate=%g noise=%g epochs=%d valratio=%g seed=%d train=%d val=%d test=%d\n",
		levels, cfg.useBias, cfg.cellMask, cfg.biasChoice, cfg.lr, cfg.margin,
		cfg.groupSize, cfg.gateScale, cfg.noiseScale, cfg.epochs, cfg.valRatio, cfg.seed,
		len(trXs), len(valXs), len(testXs))

	bestVal, testAtBestVal, bestT, bestEpoch := math.Inf(1), 0.0, 0.0, 0
	for e := 1; e <= cfg.epochs; e++ {
		t0 := time.Now()
		if err := tr.trainEpoch(trXs, trLevels); err != nil {
			return err
		}
		valLogits, err := m.allLogits(valXs, workers)
		if err != nil {
			return err
		}
		tmp, valMAE := m.bestTemperature(valLogits, valLevels)
		testLogits, err := m.allLogits(testXs, workers)
		if err != nil {
			return err
		}
		testMAE := m.maeFromLogits(testLogits, testLevels, tmp)
		if valMAE < bestVal {
			bestVal, testAtBestVal, bestT, bestEpoch = valMAE, testMAE, tmp, e
		}
		fmt.Printf("epoch %d: val MAE %.3f / test MAE %.3f (T=%.0f, best val %.3f @%d) %.1fs\n",
			e, valMAE, testMAE, tmp, bestVal, bestEpoch, time.Since(t0).Seconds())
	}
	if err := reportPreActivationStd(m, testXs, workers); err != nil {
		return err
	}
	reportBias(m)
	fmt.Printf("希望活性の極性(累計): +1 = %d / -1 = %d\n", tr.posDesired, tr.negDesired)
	fmt.Printf("BEST_VAL_MAE %.3f @epoch %d (T=%.0f) / TEST_MAE %.3f\n", bestVal, bestEpoch, bestT, testAtBestVal)

	// --- 復号方式の比較 ---
	// 復号は学習済みモデルへの後処理なので、同じモデルで全方式を測れる。
	// パラメータを持つ方式は検証集合で選び、テスト集合は選択に使わない。
	valLogits, err := m.allLogits(valXs, workers)
	if err != nil {
		return err
	}
	testLogits, err := m.allLogits(testXs, workers)
	if err != nil {
		return err
	}
	totalBits := m.prototypes[0].Rows() * m.prototypes[0].Cols()
	maxRate, minRate, rateRange := logitSpread(testLogits, totalBits)
	fmt.Printf("\n一致率の散らばり(テスト集合): 最大 %.4f / 最小 %.4f / 幅 %.4f (総ビット %d)\n",
		maxRate, minRate, rateRange, totalBits)
	fmt.Printf("%-42s %9s %10s %10s\n", "復号方式", "検証MAE", "テストMAE", "パラメータ")
	for _, spec := range decodeSpecs() {
		v, t, prm := spec.evaluate(valLogits, valLevels, testLogits, testLevels, m.values)
		if spec.tune == nil {
			fmt.Printf("%-42s %9.3f %10.3f %10s\n", spec.name, v, t, "-")
		} else {
			fmt.Printf("%-42s %9.3f %10.3f %10.0f\n", spec.name, v, t, prm)
		}
	}

	// --- 点灯数による復号(プロトタイプとの距離を一切計算しない) ---
	valCounts, err := m.allCounts(valXs, workers)
	if err != nil {
		return err
	}
	testCounts, err := m.allCounts(testXs, workers)
	if err != nil {
		return err
	}
	lo, hi := m.values[0], m.values[len(m.values)-1]
	rawA := (hi - lo) / float64(totalBits)
	fitA, fitB := countLinearFit(valCounts, valLevels, m.values)
	meanPred, meanTrue := countStats(testCounts, testLevels, m.values, totalBits)

	fmt.Printf("%-42s %9.3f %10.3f %10s\n", "点灯数から直接(校正なし)",
		countMAE(valCounts, valLevels, m.values, rawA, lo),
		countMAE(testCounts, testLevels, m.values, rawA, lo), "-")
	fmt.Printf("%-42s %9.3f %10.3f %10s\n", "点灯数 + 検証で線形校正",
		countMAE(valCounts, valLevels, m.values, fitA, fitB),
		countMAE(testCounts, testLevels, m.values, fitA, fitB), "a,b")
	fmt.Printf("  診断: 校正なしの予測平均 %.2f / 真値の平均 %.2f (ずれ %+.2f)\n",
		meanPred, meanTrue, meanPred-meanTrue)
	fmt.Printf("  校正: 素の傾き %.5f -> 検証での最小二乗 a=%.5f b=%+.2f\n", rawA, fitA, fitB)
	return nil
}

func main() {
	var (
		task       = flag.String("task", "classify", "classify | regress")
		dsName     = flag.String("dataset", "mnist", "mnist | fashion (classify時)")
		useBias    = flag.Bool("bias", true, "学習する整数バイアスを使う")
		cellMask   = flag.Bool("cellmask", false, "見本にセルマスクを適用する")
		biasChoice = flag.Float64("biaschoice", 0.5, "Update時にバイアス側が選ばれる確率")
		lr         = flag.Float64("lr", 0.1, "重み側の確率的学習率")
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

	cfg := config{
		useBias:    *useBias,
		cellMask:   *cellMask,
		biasChoice: float32(*biasChoice),
		lr:         float32(*lr),
		margin:     float32(*margin),
		groupSize:  *groupSize,
		gateScale:  float32(*gateScale),
		noiseScale: float32(*noiseScale),
		epochs:     *epochs,
		batch:      *batch,
		valRatio:   *valRatio,
		seed:       *seed,
	}

	var err error
	if *task == "regress" {
		err = runRegression(cfg, workers)
	} else {
		err = runClassification(*dsName, cfg, workers)
	}
	if err != nil {
		log.Fatal(err)
	}
}
