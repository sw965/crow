// glulab: BEP の層を二値の乗算ユニットへ拡張する実験プログラム。
//
// ../layer.go の層は a = sign(W x) である。ここでは枝を2本にして
//
//	a = sign(W_u x) ⊙ sign(W_g x)
//
// とする。±1 同士の積なので a は ±1 のままで、BEP が前提とする
// 超立方体上の最適化は壊れない。
//
// SwiGLU の類推で語られることがあるが、実体は違う。sign(p)·sign(q) = sign(p·q) なので
// この層は「2つの超平面の XNOR」を1ユニットで計算する。SwiGLU の強さは連続値ゲートが
// 情報量を段階的に調整することから来るが、±1 の積にはその成分が無い。増えるのは
// 各ユニットの表現力(単一パーセプトロンでは表せない線形分離不可能な関数を表せる)であり、
// 比較すべき対照は「ゲートの有無」ではなく幅とパラメータ数になる。
//
// 枝の目標値は、他の枝を固定したときに一意に決まる。
//
//	u* = a* ⊙ g,   g* = a* ⊙ u
//
// 重要なのは、この2つを同時に適用してはならないことである。両枝を反転させると
// XNOR(¬u, ¬g) = XNOR(u, g) で出力が戻ってしまう。したがって1ニューロンにつき
// 補正を流す枝は必ず1本に絞る(-route)。
//
//	cheap : |z| の小さい方の枝(反転させやすい方)を直す
//	alt   : バッチごとに枝を交互に切り替える
//	both  : 両枝を同時に直す。上記のとおり破綻するはずの対照
//
// 対照実験(-arch):
//
//	plain : 784 -> 512 -> 1024 の通常BEP         (925,696 重み)
//	wide  : 784 -> 1024 -> 1024 の通常BEP        (1,851,392 重み)
//	gated : 784 -> 512 -> 1024 の乗算ユニット     (1,851,392 重み)
//
// wide と gated は重み数が厳密に一致する。gated の優位がパラメータ増によるものか
// 乗算構造によるものかを分離するための設計である。
//
// 逆伝播のゲートは既定で open(遮断しない)。../gatelab の実測で、|z| によるゲートは
// MNIST -2.13pt / Fashion -0.79pt と有害だったため、既知の劣る土台の上に
// 新機構を積まないようにする。-gate abs で論文どおりの条件にもできる。
//
// crow / omw のライブラリコードは変更していない。
//
// 実行例:
//
//	go run . -arch plain
//	go run . -arch wide
//	go run . -arch gated -route cheap
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
	wordBits   = 64
)

// noiseKind は活性前値へ加えるノイズの分布。
//
// norm は ../layer.go と同じガウス(randx.IntNorm)で、標準偏差が float32 のため
// 学習ループに浮動小数が残る。uniform は整数一様で、半幅 w の整数を引くだけなので
// 浮動小数を必要としない。ノイズの役割が「決定論的な二値ネットが同じ誤りを
// 繰り返すのを崩す」ことなら、分布がガウスである必要は薄いという仮説を試す。
//
// 標準偏差を揃えるため、uniform の半幅は w = scale * sqrt(3) * sqrt(fanIn) とする
// (一様分布 [-w, w] の標準偏差は w/sqrt(3))。
type noiseKind int

const (
	noiseNorm noiseKind = iota
	noiseUniform
)

func parseNoiseKind(s string) (noiseKind, error) {
	switch s {
	case "norm":
		return noiseNorm, nil
	case "uniform":
		return noiseUniform, nil
	}
	return 0, fmt.Errorf("未知のノイズ種別: %s", s)
}

func (n noiseKind) String() string {
	return [...]string{"norm", "uniform"}[n]
}

type routeMode int

const (
	routeCheap routeMode = iota
	routeAlt
	routeBoth
)

func parseRoute(s string) (routeMode, error) {
	switch s {
	case "cheap":
		return routeCheap, nil
	case "alt":
		return routeAlt, nil
	case "both":
		return routeBoth, nil
	}
	return 0, fmt.Errorf("未知の振り分けモード: %s", s)
}

func (r routeMode) String() string {
	return [...]string{"cheap", "alt", "both"}[r]
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// ---------------------------------------------------------------------------
// デルタ
// ---------------------------------------------------------------------------

type delta struct {
	// w は枝ごとの重みデルタ。plain は1本、gated は2本。
	w [][]int16

	// 枝が同符号だった割合。1に張り付いたら2枝が縮退している。
	branchAgree int64
	outPos      int64
	seen        int64
}

func newDelta(branches, wRows, wCols int) *delta {
	w := make([][]int16, branches)
	for b := range w {
		w[b] = make([]int16, wRows*wCols)
	}
	return &delta{w: w}
}

func (d *delta) clear() {
	for _, w := range d.w {
		clear(w)
	}
	d.branchAgree = 0
	d.outPos = 0
	d.seen = 0
}

func (d *delta) add(other *delta) {
	for b, w := range other.w {
		dst := d.w[b]
		for i, v := range w {
			dst[i] += v
		}
	}
	d.branchAgree += other.branchAgree
	d.outPos += other.outPos
	d.seen += other.seen
}

func (d *delta) sign() {
	for _, w := range d.w {
		for i, v := range w {
			w[i] = int16(cmp.Compare(v, 0))
		}
	}
}

// ---------------------------------------------------------------------------
// 層
// ---------------------------------------------------------------------------

type dense struct {
	// w は枝ごとの重み。len(w) == 1 なら ../layer.go と同じ通常の層になる。
	w  []*bitsx.Matrix
	wt []*bitsx.Matrix
	h  [][]int8

	noiseStd  float32
	noiseKind noiseKind

	gateOpen     bool
	gateScale    float32
	absThreshold int

	route     routeMode
	groupSize int

	// lrDen > 0 なら更新確率を有理数 lrNum/lrDen の Bernoulli で引く。
	// 0 なら従来どおり rng.Float32() と float の lr を比較する。
	lrNum, lrDen int

	// altBranch は route=alt のときに今回補正する枝。バッチごとに切り替える。
	altBranch int

	epochAgree int64
	epochPos   int64
	epochSeen  int64
}

func newDense(branches, wRows, wCols int, rng *rand.Rand) (*dense, error) {
	d := &dense{
		w:         make([]*bitsx.Matrix, branches),
		wt:        make([]*bitsx.Matrix, branches),
		h:         make([][]int8, branches),
		noiseStd:  float32(math.Sqrt(float64(wCols))),
		gateOpen:  true,
		gateScale: 1.0,
		groupSize: 4,
	}
	for b := range branches {
		w, err := bitsx.NewRandMatrix(wRows, wCols, rng)
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
		d.w[b], d.wt[b], d.h[b] = w, wt, h
	}
	d.absThreshold = int(d.noiseStd)
	return d, nil
}

func (d *dense) branches() int { return len(d.w) }

func (d *dense) fanIn() int { return d.w[0].Cols() }

func (d *dense) yCols() int { return d.w[0].Rows() }

func (d *dense) newDelta() *delta {
	return newDelta(d.branches(), d.w[0].Rows(), d.w[0].Cols())
}

func (d *dense) outputShape(xRows, xCols int) (int, int, error) {
	if xCols != d.fanIn() {
		return 0, 0, fmt.Errorf("入力の列数が不一致: xCols = %d, W.Cols = %d", xCols, d.fanIn())
	}
	return xRows, d.yCols(), nil
}

func (d *dense) applyGateScale() {
	d.absThreshold = int(d.gateScale * float32(math.Sqrt(float64(d.fanIn()))))
}

// preActivation は枝ごとの z を返す。
func (d *dense) preActivation(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) ([][]int, int, int, error) {
	fanIn := d.fanIn()
	std := noiseScale * d.noiseStd
	// 一様ノイズの半幅。ガウスと標準偏差を揃える。
	halfWidth := int(float64(std) * math.Sqrt(3))
	useUniform := d.noiseKind == noiseUniform && halfWidth > 0
	zs := make([][]int, d.branches())
	for b, w := range d.w {
		u, err := x.Dot(w)
		if err != nil {
			return nil, 0, 0, err
		}
		z := make([]int, len(u))
		for i, count := range u {
			zi := 2*count - fanIn
			switch {
			case useUniform:
				zi += rng.IntN(2*halfWidth+1) - halfWidth
			case std > 0:
				noise, err := randx.IntNorm(-fanIn, fanIn, 0, std, rng)
				if err != nil {
					return nil, 0, 0, err
				}
				zi += noise
			}
			z[i] = zi
		}
		zs[b] = z
	}
	return zs, x.Rows(), d.yCols(), nil
}

// signBit は z >= 0 を 1 とするビット表現を返す。
func signBit(z int) uint64 {
	if z >= 0 {
		return 1
	}
	return 0
}

// outputBit は枝の符号ビットの XNOR。枝が1本ならその符号そのものになる。
func outputBit(zs [][]int, idx int) uint64 {
	out := uint64(1)
	for _, z := range zs {
		// XNOR の単位元は 1 なので、1 から畳み込むと枝1本の場合も一致する。
		out = 1 &^ (out ^ signBit(z[idx]))
	}
	return out
}

type backward func(t *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error)

func (d *dense) forward(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) (*bitsx.Matrix, backward, error) {
	zs, yRows, yCols, err := d.preActivation(x, noiseScale, rng)
	if err != nil {
		return nil, nil, err
	}

	y, err := bitsx.NewZerosMatrix(yRows, yCols)
	if err != nil {
		return nil, nil, err
	}
	if err := y.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		var word uint64
		if err := ctx.ScanBits(func(i, col, colT int) error {
			if outputBit(zs, ctx.GlobalStart+i) == 1 {
				word |= 1 << uint64(i)
			}
			return nil
		}); err != nil {
			return err
		}
		return y.SetWord(ctx.WordIndex, word)
	}); err != nil {
		return nil, nil, err
	}

	bw := func(t *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error) {
		if err := t.ValidateSameShape(y); err != nil {
			return nil, err
		}
		nb := d.branches()

		// branchTarget[b] は枝bへ渡す希望活性。既定は現在の符号で、
		// 補正を流す枝だけを反転させる。両枝を同時に反転させると
		// XNOR が元に戻るため、ここで1本に絞ることが正しさの要件になる。
		branchTarget := make([]*bitsx.Matrix, nb)
		for b := range nb {
			m, err := bitsx.NewZerosMatrix(yRows, yCols)
			if err != nil {
				return nil, err
			}
			branchTarget[b] = m
		}
		keepGate, err := bitsx.NewZerosMatrix(yRows, yCols)
		if err != nil {
			return nil, err
		}

		type wordMismatch struct {
			conf   int
			branch int
			tBit   uint64
			col    int
			idx    int
		}
		mismatches := make([]wordMismatch, 0, wordBits)

		err = t.ScanRowsWord(nil, func(tCtx bitsx.MatrixWordContext) error {
			tWord, err := t.Word(tCtx.WordIndex)
			if err != nil {
				return err
			}
			mismatches = mismatches[:0]
			branchWords := make([]uint64, nb)
			var keepGateWord uint64

			if err := tCtx.ScanBits(func(i, col, colT int) error {
				idx := tCtx.GlobalStart + i

				conf := math.MaxInt
				cheap := 0
				agree := true
				first := signBit(zs[0][idx])
				for b, z := range zs {
					a := absInt(z[idx])
					if a < conf {
						conf, cheap = a, b
					}
					if signBit(z[idx]) != first {
						agree = false
					}
					if signBit(z[idx]) == 1 {
						branchWords[b] |= 1 << uint64(i)
					}
				}
				if agree {
					dl.branchAgree++
				}
				dl.seen++

				yBit := outputBit(zs, idx)
				if yBit == 1 {
					dl.outPos++
				}

				if d.gateOpen || conf <= d.absThreshold {
					keepGateWord |= 1 << uint64(i)
				}

				tBit := tWord >> uint64(i) & 1
				if tBit == yBit {
					return nil
				}

				target := cheap
				switch d.route {
				case routeAlt:
					target = d.altBranch % nb
				case routeBoth:
					target = -1
				}
				mismatches = append(mismatches,
					wordMismatch{conf: conf, branch: target, tBit: tBit, col: col, idx: idx})
				return nil
			}); err != nil {
				return err
			}

			// 不一致ニューロンは、選ばれた枝の符号を反転させることで出力が t に一致する。
			for _, mm := range mismatches {
				bit := uint64(1) << uint64(mm.col-tCtx.ColStart)
				if mm.branch < 0 {
					for b := range nb {
						branchWords[b] ^= bit
					}
				} else {
					branchWords[mm.branch] ^= bit
				}
			}

			for b := range nb {
				if err := branchTarget[b].SetWord(tCtx.WordIndex, branchWords[b]); err != nil {
					return err
				}
			}
			if err := keepGate.SetWord(tCtx.WordIndex, keepGateWord); err != nil {
				return err
			}

			slices.SortFunc(mismatches, func(a, b wordMismatch) int {
				return cmp.Compare(a.conf, b.conf)
			})
			span := tCtx.ColEnd - tCtx.ColStart
			updateK := min(max(span/d.groupSize, 1), len(mismatches))

			for _, mm := range mismatches[:updateK] {
				targets := []int{mm.branch}
				if mm.branch < 0 {
					targets = make([]int, nb)
					for b := range nb {
						targets[b] = b
					}
				}
				for _, b := range targets {
					// 枝bへの希望ビットは、反転後の branchWords から読み直す。
					word, err := branchTarget[b].Word(tCtx.WordIndex)
					if err != nil {
						return err
					}
					want := word >> uint64(mm.col-tCtx.ColStart) & 1
					deltaRow := dl.w[b][mm.col*d.fanIn() : (mm.col+1)*d.fanIn()]
					if err := x.ScanRowsWord([]int{tCtx.Row}, func(xCtx bitsx.MatrixWordContext) error {
						xWord, err := x.Word(xCtx.WordIndex)
						if err != nil {
							return err
						}
						dw := deltaRow[xCtx.ColStart:xCtx.ColEnd]
						for j := range dw {
							xBit := xWord >> uint(j) & 1
							dw[j] += int16(1 - 2*int(xBit^want))
						}
						return nil
					}); err != nil {
						return err
					}
				}
			}
			return nil
		})
		if err != nil {
			return nil, err
		}

		// 前層への希望活性は、全枝の票を整数で足してから符号を取る。
		fanIn := d.fanIn()
		sum := make([]int, yRows*fanIn)
		for b := range nb {
			raw, err := d.wt[b].DotTernary(branchTarget[b], keepGate)
			if err != nil {
				return nil, err
			}
			for i, v := range raw {
				sum[i] += v
			}
		}
		nextT, err := bitsx.NewZerosMatrix(yRows, fanIn)
		if err != nil {
			return nil, err
		}
		if err := nextT.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
			var word uint64
			if err := ctx.ScanBits(func(i, col, colT int) error {
				if sum[colT] >= 0 {
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
	zs, yRows, yCols, err := d.preActivation(x, 0, nil)
	if err != nil {
		return nil, err
	}
	y, err := bitsx.NewZerosMatrix(yRows, yCols)
	if err != nil {
		return nil, err
	}
	if err := y.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		var word uint64
		if err := ctx.ScanBits(func(i, col, colT int) error {
			if outputBit(zs, ctx.GlobalStart+i) == 1 {
				word |= 1 << uint64(i)
			}
			return nil
		}); err != nil {
			return err
		}
		return y.SetWord(ctx.WordIndex, word)
	}); err != nil {
		return nil, err
	}
	return y, nil
}

func (d *dense) absorbStats(dl *delta) {
	d.epochAgree += dl.branchAgree
	d.epochPos += dl.outPos
	d.epochSeen += dl.seen
	d.altBranch++
}

func (d *dense) resetEpochStats() {
	d.epochAgree = 0
	d.epochPos = 0
	d.epochSeen = 0
}

// update は ../layer.go の Update と同じ手順を枝ごとに行う。
func (d *dense) update(dl *delta, lr float32, rng *rand.Rand) error {
	for b, w := range d.w {
		hAll := d.h[b]
		dAll := dl.w[b]
		wt := d.wt[b]
		if err := w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
			hWord := hAll[ctx.GlobalStart:ctx.GlobalEnd]
			dWord := dAll[ctx.GlobalStart:ctx.GlobalEnd]
			var flips uint64
			if err := ctx.ScanBits(func(i, col, colT int) error {
				if d.lrDen > 0 {
					if rng.IntN(d.lrDen) >= d.lrNum {
						return nil
					}
				} else if rng.Float32() > lr {
					return nil
				}
				old := hWord[i]
				newVal := int(old) + int(dWord[i])
				clipped := int8(max(math.MinInt8, min(newVal, math.MaxInt8)))
				hWord[i] = clipped
				if (old >= 0) != (clipped >= 0) {
					flips |= 1 << uint64(i)
					// 逆伝播は wt を使うため、可視重みの反転を転置側へも反映する。
					return wt.Toggle(col, ctx.Row)
				}
				return nil
			}); err != nil {
				return err
			}
			old, err := w.Word(ctx.WordIndex)
			if err != nil {
				return err
			}
			return w.SetWord(ctx.WordIndex, old^flips)
		}); err != nil {
			return err
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// モデル
// ---------------------------------------------------------------------------

type model struct {
	layers     []*dense
	prototypes bitsx.Matrices
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

func (m *model) appendLayer(branches, wRows int, rng *rand.Rand) error {
	wCols := m.xCols
	if len(m.layers) > 0 {
		_, c, err := m.outputShape()
		if err != nil {
			return err
		}
		wCols = c
	}
	l, err := newDense(branches, wRows, wCols, rng)
	if err != nil {
		return err
	}
	m.layers = append(m.layers, l)
	return nil
}

func (m *model) weightCount() int {
	total := 0
	for _, l := range m.layers {
		total += l.branches() * l.yCols() * l.fanIn()
	}
	return total
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

// ---------------------------------------------------------------------------
// 学習
// ---------------------------------------------------------------------------

type trainer struct {
	model         *model
	miniBatchSize int
	lr            float32
	margin        float32
	// marginMode は margin を何に対する比率とみなすか。
	// total は出力の総ビット数の半分(旧 crow 本体と同じ)、
	// mindist はプロトタイプ間の最小ハミング距離(現 crow 本体と同じ)。
	marginMode   string
	noiseScale   float32
	workerRNGs   []*rand.Rand
	shuffleRNG   *rand.Rand
	updateRNG    *rand.Rand
	workerDeltas [][]*delta
	aggregated   []*delta
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
		model: m, miniBatchSize: 1024, lr: 0.1, margin: 0.5, marginMode: "total", noiseScale: 0.5,
		workerRNGs:   rngs,
		shuffleRNG:   rand.New(rand.NewPCG(seed, 0xD1B54A32D192ED03)),
		updateRNG:    rand.New(rand.NewPCG(seed, 0xA24BAED4963EE407)),
		workerDeltas: wd, aggregated: agg,
	}, nil
}

func minPrototypeHammingDistance(protos bitsx.Matrices) (int, error) {
	if len(protos) < 2 {
		return 0, nil
	}
	minDist := math.MaxInt
	for i := range protos {
		for j := i + 1; j < len(protos); j++ {
			d, err := protos[i].HammingDistance(protos[j])
			if err != nil {
				return 0, err
			}
			minDist = min(minDist, d)
		}
	}
	return minDist, nil
}

// marginBits は margin を一致ビット数の差へ変換する。
func (t *trainer) marginBits() (int, error) {
	protos := t.model.prototypes
	switch t.marginMode {
	case "total":
		totalBits := protos[0].Rows() * protos[0].Cols()
		return int(float32(totalBits) * t.margin / 2), nil
	case "mindist":
		d, err := minPrototypeHammingDistance(protos)
		if err != nil {
			return 0, err
		}
		return int(t.margin * float32(d)), nil
	}
	return 0, fmt.Errorf("未知のmarginMode: %s", t.marginMode)
}

func satisfiesUpdateCriterion(y *bitsx.Matrix, label int, protos bitsx.Matrices, marginBits int) (bool, error) {
	t := protos[label]
	yMismatch, err := y.HammingDistance(t)
	if err != nil {
		return false, err
	}
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
	marginBits, err := t.marginBits()
	if err != nil {
		return err
	}

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
			should, err := satisfiesUpdateCriterion(y, label, t.model.prototypes, marginBits)
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
			d.sign()
		}

		for li, layer := range t.model.layers {
			layer.absorbStats(t.aggregated[li])
			if err := layer.update(t.aggregated[li], t.lr, t.updateRNG); err != nil {
				return err
			}
		}
	}
	return nil
}

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

// reportBranchStats は2枝が縮退していないかを出す。一致率が1に近づくと
// sign(z_u) == sign(z_g) となり、出力が定数に張り付いて乗算の意味が消える。
func reportBranchStats(m *model) {
	for li, l := range m.layers {
		if l.epochSeen == 0 {
			continue
		}
		fmt.Printf("  層%d 枝数 %d / 枝の符号一致率 %.3f / 出力+1率 %.3f\n",
			li, l.branches(),
			float64(l.epochAgree)/float64(l.epochSeen),
			float64(l.epochPos)/float64(l.epochSeen))
	}
}

// ---------------------------------------------------------------------------
// 実行
// ---------------------------------------------------------------------------

type config struct {
	dataset    string
	arch       string
	route      routeMode
	noiseKind  noiseKind
	lrNum      int
	lrDen      int
	gateOpen   bool
	gateScale  float32
	lr         float32
	margin     float32
	marginMode string
	groupSize  int
	noiseScale float32
	epochs     int
	batch      int
	valRatio   float64
	seed       uint64
}

// archSpec は arch 名から (枝数, 各層の出力幅) を返す。
// wide と gated は重み数が一致する。
func archSpec(arch string) (int, []int, error) {
	switch arch {
	case "plain":
		return 1, []int{512, 1024}, nil
	case "wide":
		return 1, []int{1024, 1024}, nil
	case "wider1536":
		return 1, []int{1536, 1024}, nil
	case "wider2048":
		return 1, []int{2048, 1024}, nil
	case "wider2560":
		return 1, []int{2560, 1024}, nil
	case "gated":
		return 2, []int{512, 1024}, nil
	}
	return 0, nil, fmt.Errorf("未知のarch: %s", arch)
}

func newModel(cfg config, rng *rand.Rand) (*model, error) {
	branches, widths, err := archSpec(cfg.arch)
	if err != nil {
		return nil, err
	}
	m := &model{xRows: 1, xCols: 784}
	for _, wRows := range widths {
		if err := m.appendLayer(branches, wRows, rng); err != nil {
			return nil, err
		}
	}
	yRows, yCols, err := m.outputShape()
	if err != nil {
		return nil, err
	}
	totalBits := numClasses * yRows * yCols
	iters := numClasses * int(float64(totalBits)*math.Log(float64(totalBits)))
	protos, err := bitsx.NewETFMatrices(numClasses, yRows, yCols, iters, rng)
	if err != nil {
		return nil, err
	}
	m.prototypes = protos
	for _, l := range m.layers {
		l.route = cfg.route
		l.noiseKind = cfg.noiseKind
		l.lrNum, l.lrDen = cfg.lrNum, cfg.lrDen
		l.gateOpen = cfg.gateOpen
		l.gateScale = cfg.gateScale
		l.groupSize = cfg.groupSize
		l.applyGateScale()
	}
	return m, nil
}

func run(cfg config, workers int) error {
	var ds dataset.Binary[int]
	var err error
	if cfg.dataset == "fashion" {
		ds, err = dataset.LoadFashionMNIST(nil)
	} else {
		ds, err = dataset.LoadMNIST(nil)
	}
	if err != nil {
		return err
	}

	rng := rand.New(rand.NewPCG(cfg.seed, cfg.seed+1))
	m, err := newModel(cfg, rng)
	if err != nil {
		return err
	}

	tr, err := newTrainer(m, workers, cfg.seed)
	if err != nil {
		return err
	}
	tr.miniBatchSize = cfg.batch
	tr.lr = cfg.lr
	tr.margin = cfg.margin
	tr.marginMode = cfg.marginMode
	tr.noiseScale = cfg.noiseScale

	splitRNG := rand.New(rand.NewPCG(cfg.seed, 0xC2B2AE3D27D4EB4F))
	trainXs, trainLabels, valXs, valLabels, err := splitTrainValidation(
		ds.TrainInputs, ds.TrainLabels, cfg.valRatio, splitRNG)
	if err != nil {
		return err
	}

	gateName := "open"
	if !cfg.gateOpen {
		gateName = "abs"
	}
	fmt.Printf("dataset=%s arch=%s route=%s noisekind=%s gate=%s weights=%d lrnum=%d lrden=%d lr=%g margin=%g marginmode=%s gsize=%d noise=%g epochs=%d seed=%d train=%d val=%d test=%d\n",
		cfg.dataset, cfg.arch, cfg.route, cfg.noiseKind, gateName, m.weightCount(),
		cfg.lrNum, cfg.lrDen, cfg.lr, cfg.margin, cfg.marginMode, cfg.groupSize, cfg.noiseScale, cfg.epochs, cfg.seed,
		len(trainXs), len(valXs), len(ds.TestInputs))

	bestVal, testAtBestVal, bestEpoch := -1.0, 0.0, 0
	for e := 1; e <= cfg.epochs; e++ {
		t0 := time.Now()
		for _, l := range m.layers {
			l.resetEpochStats()
		}
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
	reportBranchStats(m)
	fmt.Printf("BEST_VAL %.4f @epoch %d / TEST %.4f\n", bestVal, bestEpoch, testAtBestVal)
	return nil
}

func main() {
	var (
		dsName     = flag.String("dataset", "mnist", "mnist | fashion")
		arch       = flag.String("arch", "plain", "plain | wide | wider1536 | wider2048 | wider2560 | gated")
		routeName  = flag.String("route", "cheap", "補正を流す枝 cheap | alt | both")
		noiseName  = flag.String("noisekind", "norm", "ノイズ分布 norm | uniform")
		gateName   = flag.String("gate", "open", "逆伝播ゲート open | abs")
		gateScale  = flag.Float64("gscale", 1.0, "abs時のしきい値スケール(√fanIn倍)")
		lr         = flag.Float64("lr", 0.1, "確率的学習率(lrdenが0のときに使う)")
		lrNum      = flag.Int("lrnum", 0, "有理数学習率の分子")
		lrDen      = flag.Int("lrden", 0, "有理数学習率の分母(0なら浮動小数のlrを使う)")
		margin     = flag.Float64("margin", 0.5, "更新判定のマージン(比率)")
		marginMode = flag.String("marginmode", "total", "marginの基準 total(総ビット数/2) | mindist(プロトタイプ間の最小距離)")
		groupSize  = flag.Int("gsize", 4, "GroupSize")
		noiseScale = flag.Float64("noise", 0.5, "NoiseStdScale")
		epochs     = flag.Int("epochs", 20, "エポック数")
		batch      = flag.Int("batch", 1024, "ミニバッチサイズ")
		valRatio   = flag.Float64("valratio", 0.1, "学習データから検証用に分ける割合")
		seed       = flag.Uint64("seed", 1, "乱数シード")
		threads    = flag.Int("threads", 0, "ワーカー数 (0 = NumCPU)")
	)
	flag.Parse()

	route, err := parseRoute(*routeName)
	if err != nil {
		log.Fatal(err)
	}
	nk, err := parseNoiseKind(*noiseName)
	if err != nil {
		log.Fatal(err)
	}
	if *gateName != "open" && *gateName != "abs" {
		log.Fatalf("未知のゲート: %s", *gateName)
	}
	if _, _, err := archSpec(*arch); err != nil {
		log.Fatal(err)
	}

	workers := *threads
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	cfg := config{
		dataset:    *dsName,
		arch:       *arch,
		route:      route,
		noiseKind:  nk,
		lrNum:      *lrNum,
		lrDen:      *lrDen,
		gateOpen:   *gateName == "open",
		gateScale:  float32(*gateScale),
		lr:         float32(*lr),
		margin:     float32(*margin),
		marginMode: *marginMode,
		groupSize:  *groupSize,
		noiseScale: float32(*noiseScale),
		epochs:     *epochs,
		batch:      *batch,
		valRatio:   *valRatio,
		seed:       *seed,
	}

	if err := run(cfg, workers); err != nil {
		log.Fatal(err)
	}
}
