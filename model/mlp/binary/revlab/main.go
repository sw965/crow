// revlab は、情報を落とさない可逆二値バックボーンを BEP の前段に置く実験プログラム。
//
// 可逆ブロックは 1024 ビットの状態を半分に分け、
//
//	q      = sign(W * left + b)
//	toggled = right * q       // ±1 の積。ビット表現では XNOR
//	output  = (toggled, left) // 固定の左右交換
//
// とする。逆変換は left=output.right,
// right=output.left*q なので、重みの値によらず厳密に可逆である。
// 最後に通常の非可逆 Dense を一層だけ置き、固定 ETF プロトタイプへ凝集させる。
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
	hInitAbs   = 4
	numClasses = 10
)

type delta struct {
	w []int16
	b []int16
}

func newDelta(rows, cols int) *delta {
	return &delta{w: make([]int16, rows*cols), b: make([]int16, rows)}
}

func (d *delta) clear() {
	clear(d.w)
	clear(d.b)
}

func (d *delta) add(other *delta) {
	for i, v := range other.w {
		d.w[i] += v
	}
	for i, v := range other.b {
		d.b[i] += v
	}
}

func (d *delta) sign() {
	for i, v := range d.w {
		d.w[i] = int16(cmp.Compare(v, 0))
	}
	for i, v := range d.b {
		d.b[i] = int16(cmp.Compare(v, 0))
	}
}

type dense struct {
	w  *bitsx.Matrix
	wt *bitsx.Matrix
	h  []int8

	bias    []int32
	biasMax int32

	gateBase   int
	noiseStd   float32
	gateScale  float32
	groupSize  int
	useBias    bool
	biasChoice float32
}

func newDense(rows, cols int, useBias bool, biasChoice float32, rng *rand.Rand) (*dense, error) {
	w, err := bitsx.NewRandMatrix(rows, cols, 0, rng)
	if err != nil {
		return nil, err
	}
	wt, err := w.Transpose()
	if err != nil {
		return nil, err
	}
	h := make([]int8, rows*cols)
	if err := w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		word, err := w.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		hw := h[ctx.GlobalStart:ctx.GlobalEnd]
		return ctx.ScanBits(func(i, col, colT int) error {
			if word>>uint(i)&1 == 1 {
				hw[i] = hInitAbs
			} else {
				hw[i] = -hInitAbs
			}
			return nil
		})
	}); err != nil {
		return nil, err
	}
	std := float32(math.Sqrt(float64(cols)))
	return &dense{
		w: w, wt: wt, h: h,
		bias: make([]int32, rows), biasMax: int32(cols),
		gateBase: int(std), noiseStd: std, gateScale: 1, groupSize: 4,
		useBias: useBias, biasChoice: biasChoice,
	}, nil
}

func (d *dense) newDelta() *delta { return newDelta(d.w.Rows(), d.w.Cols()) }

type denseTrace struct {
	x *bitsx.Matrix
	y *bitsx.Matrix
	z []int
}

func (d *dense) preActivation(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) ([]int, error) {
	u, err := x.Dot(d.w)
	if err != nil {
		return nil, err
	}
	rows := d.w.Rows()
	fanIn := d.w.Cols()
	std := noiseScale * d.noiseStd
	z := make([]int, len(u))
	for i, count := range u {
		zi := 2*count - fanIn
		if d.useBias {
			zi += int(d.bias[i%rows])
		}
		if std > 0 {
			noise, err := randx.IntNorm(-fanIn, fanIn, 0, std, rng)
			if err != nil {
				return nil, err
			}
			zi += noise
		}
		z[i] = zi
	}
	return z, nil
}

func (d *dense) forward(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) (*bitsx.Matrix, *denseTrace, error) {
	z, err := d.preActivation(x, noiseScale, rng)
	if err != nil {
		return nil, nil, err
	}
	y, err := bitsx.NewSignMatrix(x.Rows(), d.w.Rows(), z)
	if err != nil {
		return nil, nil, err
	}
	return y, &denseTrace{x: x, y: y, z: z}, nil
}

func (d *dense) predict(x *bitsx.Matrix) (*bitsx.Matrix, error) {
	z, err := d.preActivation(x, 0, nil)
	if err != nil {
		return nil, err
	}
	return bitsx.NewSignMatrix(x.Rows(), d.w.Rows(), z)
}

type mismatch struct {
	absZ int
	bit  uint64
	col  int
}

func (d *dense) collectMismatches(trace *denseTrace, target *bitsx.Matrix, dl *delta) error {
	if err := target.ValidateSameShape(trace.y); err != nil {
		return err
	}
	return target.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		zWord := trace.z[ctx.GlobalStart:ctx.GlobalEnd]
		tWord, err := target.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		mms := make([]mismatch, 0, 64)
		if err := ctx.ScanBits(func(i, col, colT int) error {
			absZ := zWord[i]
			if absZ < 0 {
				absZ = -absZ
			}
			tBit := tWord >> uint(i) & 1
			yBit := uint64(0)
			if zWord[i] >= 0 {
				yBit = 1
			}
			if tBit != yBit {
				mms = append(mms, mismatch{absZ: absZ, bit: tBit, col: col})
			}
			return nil
		}); err != nil {
			return err
		}
		slices.SortFunc(mms, func(a, b mismatch) int { return cmp.Compare(a.absZ, b.absZ) })
		updateK := min(max(len(zWord)/d.groupSize, 1), len(mms))
		for _, mm := range mms[:updateK] {
			if mm.bit == 1 {
				dl.b[mm.col]++
			} else {
				dl.b[mm.col]--
			}
			dw := dl.w[mm.col*d.w.Cols() : (mm.col+1)*d.w.Cols()]
			if err := trace.x.ScanRowsWord([]int{ctx.Row}, func(xCtx bitsx.MatrixWordContext) error {
				xWord, err := trace.x.Word(xCtx.WordIndex)
				if err != nil {
					return err
				}
				part := dw[xCtx.ColStart:xCtx.ColEnd]
				for i := range part {
					xBit := xWord >> uint(i) & 1
					part[i] += int16(1 - 2*int(xBit^mm.bit))
				}
				return nil
			}); err != nil {
				return err
			}
		}
		return nil
	})
}

func (d *dense) backward(trace *denseTrace, target *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error) {
	if err := d.collectMismatches(trace, target, dl); err != nil {
		return nil, err
	}
	keep, err := bitsx.NewZerosMatrix(target.Rows(), target.Cols())
	if err != nil {
		return nil, err
	}
	gate := int(d.gateScale * float32(d.gateBase))
	if err := keep.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		zWord := trace.z[ctx.GlobalStart:ctx.GlobalEnd]
		var word uint64
		if err := ctx.ScanBits(func(i, col, colT int) error {
			absZ := zWord[i]
			if absZ < 0 {
				absZ = -absZ
			}
			if absZ <= gate {
				word |= 1 << uint(i)
			}
			return nil
		}); err != nil {
			return err
		}
		return keep.SetWord(ctx.WordIndex, word)
	}); err != nil {
		return nil, err
	}
	raw, err := d.wt.DotTernary(target, keep)
	if err != nil {
		return nil, err
	}
	return bitsx.NewSignMatrix(target.Rows(), d.w.Cols(), raw)
}

func (d *dense) update(dl *delta, lr float32, rng *rand.Rand) error {
	biasMode := make([]bool, d.w.Rows())
	if d.biasChoice > 0 {
		for i := range biasMode {
			biasMode[i] = rng.Float32() < d.biasChoice
		}
	}
	if d.useBias {
		for i := range biasMode {
			if biasMode[i] {
				d.bias[i] = max(-d.biasMax, min(d.bias[i]+int32(dl.b[i]), d.biasMax))
			}
		}
	}
	return d.w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		if biasMode[ctx.Row] {
			return nil
		}
		hw := d.h[ctx.GlobalStart:ctx.GlobalEnd]
		dw := dl.w[ctx.GlobalStart:ctx.GlobalEnd]
		var flips uint64
		if err := ctx.ScanBits(func(i, col, colT int) error {
			if rng.Float32() > lr {
				return nil
			}
			old := hw[i]
			newValue := int8(max(math.MinInt8, min(int(old)+int(dw[i]), math.MaxInt8)))
			hw[i] = newValue
			if (old >= 0) != (newValue >= 0) {
				flips |= 1 << uint(i)
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

// pmMul は ±1 の要素積。Matrix の 1=+1, 0=-1 表現では XNOR になる。
func pmMul(a, b *bitsx.Matrix) (*bitsx.Matrix, error) {
	if err := a.ValidateSameShape(b); err != nil {
		return nil, err
	}
	out, err := bitsx.NewZerosMatrix(a.Rows(), a.Cols())
	if err != nil {
		return nil, err
	}
	if err := out.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		aw, err := a.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		bw, err := b.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		return out.SetWord(ctx.WordIndex, ^(aw ^ bw))
	}); err != nil {
		return nil, err
	}
	return out, nil
}

func padState(x *bitsx.Matrix, cols int) (*bitsx.Matrix, error) {
	if cols < x.Cols() {
		return nil, fmt.Errorf("状態幅 %d は入力幅 %d 以上であるべき", cols, x.Cols())
	}
	out, err := bitsx.NewZerosMatrix(x.Rows(), cols)
	if err != nil {
		return nil, err
	}
	for r := range x.Rows() {
		for s := range x.Stride() {
			word, err := x.Word(r*x.Stride() + s)
			if err != nil {
				return nil, err
			}
			if err := out.SetWord(r*out.Stride()+s, word); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

func splitHalves(x *bitsx.Matrix) (*bitsx.Matrix, *bitsx.Matrix, error) {
	if x.Cols()%128 != 0 {
		return nil, nil, fmt.Errorf("状態幅は128の倍数であるべき: %d", x.Cols())
	}
	half := x.Cols() / 2
	left, err := bitsx.NewZerosMatrix(x.Rows(), half)
	if err != nil {
		return nil, nil, err
	}
	right, err := bitsx.NewZerosMatrix(x.Rows(), half)
	if err != nil {
		return nil, nil, err
	}
	words := half / 64
	for r := range x.Rows() {
		for s := range words {
			lw, err := x.Word(r*x.Stride() + s)
			if err != nil {
				return nil, nil, err
			}
			rw, err := x.Word(r*x.Stride() + words + s)
			if err != nil {
				return nil, nil, err
			}
			if err := left.SetWord(r*left.Stride()+s, lw); err != nil {
				return nil, nil, err
			}
			if err := right.SetWord(r*right.Stride()+s, rw); err != nil {
				return nil, nil, err
			}
		}
	}
	return left, right, nil
}

func joinHalves(left, right *bitsx.Matrix) (*bitsx.Matrix, error) {
	if err := left.ValidateSameShape(right); err != nil {
		return nil, err
	}
	if left.Cols()%64 != 0 {
		return nil, fmt.Errorf("半状態幅は64の倍数であるべき: %d", left.Cols())
	}
	out, err := bitsx.NewZerosMatrix(left.Rows(), left.Cols()*2)
	if err != nil {
		return nil, err
	}
	for r := range left.Rows() {
		for s := range left.Stride() {
			lw, err := left.Word(r*left.Stride() + s)
			if err != nil {
				return nil, err
			}
			rw, err := right.Word(r*right.Stride() + s)
			if err != nil {
				return nil, err
			}
			if err := out.SetWord(r*out.Stride()+s, lw); err != nil {
				return nil, err
			}
			if err := out.SetWord(r*out.Stride()+left.Stride()+s, rw); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

type coupling struct {
	f         *dense
	learnable bool
}

type couplingTrace struct {
	left   *bitsx.Matrix
	right  *bitsx.Matrix
	fTrace *denseTrace
}

func newCoupling(stateCols int, learnable, useBias bool, biasChoice float32, rng *rand.Rand) (*coupling, error) {
	if stateCols%128 != 0 {
		return nil, fmt.Errorf("状態幅は128の倍数であるべき: %d", stateCols)
	}
	half := stateCols / 2
	f, err := newDense(half, half, useBias, biasChoice, rng)
	if err != nil {
		return nil, err
	}
	return &coupling{f: f, learnable: learnable}, nil
}

func (c *coupling) forward(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) (*bitsx.Matrix, *couplingTrace, error) {
	left, right, err := splitHalves(x)
	if err != nil {
		return nil, nil, err
	}
	q, trace, err := c.f.forward(left, noiseScale, rng)
	if err != nil {
		return nil, nil, err
	}
	toggled, err := pmMul(right, q)
	if err != nil {
		return nil, nil, err
	}
	y, err := joinHalves(toggled, left)
	if err != nil {
		return nil, nil, err
	}
	return y, &couplingTrace{left: left, right: right, fTrace: trace}, nil
}

func (c *coupling) predict(x *bitsx.Matrix) (*bitsx.Matrix, error) {
	left, right, err := splitHalves(x)
	if err != nil {
		return nil, err
	}
	q, err := c.f.predict(left)
	if err != nil {
		return nil, err
	}
	toggled, err := pmMul(right, q)
	if err != nil {
		return nil, err
	}
	return joinHalves(toggled, left)
}

func (c *coupling) inverse(y *bitsx.Matrix) (*bitsx.Matrix, error) {
	toggled, left, err := splitHalves(y)
	if err != nil {
		return nil, err
	}
	q, err := c.f.predict(left)
	if err != nil {
		return nil, err
	}
	right, err := pmMul(toggled, q)
	if err != nil {
		return nil, err
	}
	return joinHalves(left, right)
}

// backward は希望出力を現在の可逆写像で厳密に逆写像する。
// 同時に、現在の right を希望 toggled へ移すための f の希望出力を局所更新へ渡す。
func (c *coupling) backward(trace *couplingTrace, target *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error) {
	targetToggled, targetLeft, err := splitHalves(target)
	if err != nil {
		return nil, err
	}
	desiredQ, err := pmMul(targetToggled, trace.right)
	if err != nil {
		return nil, err
	}
	if c.learnable {
		if err := c.f.collectMismatches(trace.fTrace, desiredQ, dl); err != nil {
			return nil, err
		}
	}
	qAtTarget, err := c.f.predict(targetLeft)
	if err != nil {
		return nil, err
	}
	targetRight, err := pmMul(targetToggled, qAtTarget)
	if err != nil {
		return nil, err
	}
	return joinHalves(targetLeft, targetRight)
}

type backward func(target *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error)

type model struct {
	mode       string
	inputCols  int
	stateCols  int
	outputCols int

	denseLayers []*dense
	blocks      []*coupling
	head        *dense
	prototypes  bitsx.Matrices
}

func newModel(mode string, inputCols, stateCols, outputCols, blockCount int, learnRev, useBias bool, biasChoice float32, seed uint64) (*model, error) {
	if mode != "dense" && mode != "project" && mode != "reversible" {
		return nil, fmt.Errorf("mode が不正: %s", mode)
	}
	weightRNG := rand.New(rand.NewPCG(seed, 0x243F6A8885A308D3))
	protoRNG := rand.New(rand.NewPCG(seed, 0x13198A2E03707344))
	m := &model{mode: mode, inputCols: inputCols, stateCols: stateCols, outputCols: outputCols}
	if mode == "dense" {
		l1, err := newDense(512, inputCols, useBias, biasChoice, weightRNG)
		if err != nil {
			return nil, err
		}
		l2, err := newDense(outputCols, 512, useBias, biasChoice, weightRNG)
		if err != nil {
			return nil, err
		}
		m.denseLayers = []*dense{l1, l2}
	} else {
		if stateCols < inputCols || stateCols%128 != 0 {
			return nil, fmt.Errorf("state は入力幅以上かつ128の倍数であるべき: input=%d state=%d", inputCols, stateCols)
		}
		if mode == "reversible" {
			for range blockCount {
				block, err := newCoupling(stateCols, learnRev, useBias, biasChoice, weightRNG)
				if err != nil {
					return nil, err
				}
				m.blocks = append(m.blocks, block)
			}
		}
		head, err := newDense(outputCols, stateCols, useBias, biasChoice, weightRNG)
		if err != nil {
			return nil, err
		}
		m.head = head
	}
	totalBits := numClasses * outputCols
	iters := 10 * int(float64(totalBits)*math.Log(float64(totalBits)))
	prototypes, err := bitsx.NewETFMatrices(numClasses, 1, outputCols, iters, protoRNG)
	if err != nil {
		return nil, err
	}
	m.prototypes = prototypes
	return m, nil
}

func (m *model) trainableLayers() []*dense {
	if m.mode == "dense" {
		return m.denseLayers
	}
	layers := make([]*dense, 0, len(m.blocks)+1)
	for _, block := range m.blocks {
		layers = append(layers, block.f)
	}
	return append(layers, m.head)
}

func (m *model) forward(x *bitsx.Matrix, noiseScale, revNoiseScale float32, rng *rand.Rand) (*bitsx.Matrix, []backward, error) {
	if m.mode == "dense" {
		bws := make([]backward, 0, len(m.denseLayers))
		for _, layer := range m.denseLayers {
			y, trace, err := layer.forward(x, noiseScale, rng)
			if err != nil {
				return nil, nil, err
			}
			bws = append(bws, func(target *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error) {
				return layer.backward(trace, target, dl)
			})
			x = y
		}
		return x, bws, nil
	}
	state, err := padState(x, m.stateCols)
	if err != nil {
		return nil, nil, err
	}
	bws := make([]backward, 0, len(m.blocks)+1)
	for _, block := range m.blocks {
		y, trace, err := block.forward(state, revNoiseScale, rng)
		if err != nil {
			return nil, nil, err
		}
		bws = append(bws, func(target *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error) {
			return block.backward(trace, target, dl)
		})
		state = y
	}
	y, trace, err := m.head.forward(state, noiseScale, rng)
	if err != nil {
		return nil, nil, err
	}
	bws = append(bws, func(target *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error) {
		return m.head.backward(trace, target, dl)
	})
	return y, bws, nil
}

func (m *model) backbone(x *bitsx.Matrix) (*bitsx.Matrix, error) {
	state, err := padState(x, m.stateCols)
	if err != nil {
		return nil, err
	}
	for _, block := range m.blocks {
		state, err = block.predict(state)
		if err != nil {
			return nil, err
		}
	}
	return state, nil
}

func (m *model) inverseBackbone(state *bitsx.Matrix) (*bitsx.Matrix, error) {
	var err error
	for _, block := range slices.Backward(m.blocks) {
		state, err = block.inverse(state)
		if err != nil {
			return nil, err
		}
	}
	return state, nil
}

func (m *model) predict(x *bitsx.Matrix) (*bitsx.Matrix, error) {
	if m.mode == "dense" {
		var err error
		for _, layer := range m.denseLayers {
			x, err = layer.predict(x)
			if err != nil {
				return nil, err
			}
		}
		return x, nil
	}
	state, err := m.backbone(x)
	if err != nil {
		return nil, err
	}
	return m.head.predict(state)
}

func logits(y *bitsx.Matrix, prototypes bitsx.Matrices) ([]int, error) {
	total := y.Rows() * y.Cols()
	out := make([]int, len(prototypes))
	for i, p := range prototypes {
		d, err := y.HammingDistance(p)
		if err != nil {
			return nil, err
		}
		out[i] = total - d
	}
	return out, nil
}

func satisfiesUpdateCriterion(y *bitsx.Matrix, label int, prototypes bitsx.Matrices, margin float32) (bool, error) {
	lg, err := logits(y, prototypes)
	if err != nil {
		return false, err
	}
	marginBits := int(float32(y.Rows()*y.Cols()) * margin / 2)
	for c, score := range lg {
		if c != label && score-lg[label] > -marginBits {
			return true, nil
		}
	}
	return false, nil
}

func (m *model) accuracy(xs bitsx.Matrices, labels []int, workers int) (float64, error) {
	correct := make([]int, workers)
	err := parallel.For(len(xs), workers, func(workerID, i int) error {
		y, err := m.predict(xs[i])
		if err != nil {
			return err
		}
		lg, err := logits(y, m.prototypes)
		if err != nil {
			return err
		}
		best := 0
		for c := 1; c < len(lg); c++ {
			if lg[c] > lg[best] {
				best = c
			}
		}
		if best == labels[i] {
			correct[workerID]++
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	total := 0
	for _, count := range correct {
		total += count
	}
	return float64(total) / float64(len(xs)), nil
}

type trainer struct {
	model         *model
	miniBatchSize int
	lr            float32
	margin        float32
	noiseScale    float32
	revNoiseScale float32

	workerRNGs []*rand.Rand
	shuffleRNG *rand.Rand
	updateRNGs []*rand.Rand

	workerDeltas [][]*delta
	aggregated   []*delta
}

func newTrainer(m *model, workers int, seed uint64) (*trainer, error) {
	if workers <= 0 {
		return nil, errors.New("workers は1以上であるべき")
	}
	layers := m.trainableLayers()
	workerDeltas := make([][]*delta, workers)
	workerRNGs := make([]*rand.Rand, workers)
	for w := range workers {
		workerRNGs[w] = rand.New(rand.NewPCG(seed, 0x9E3779B97F4A7C15+uint64(w)))
		workerDeltas[w] = make([]*delta, len(layers))
		for i, layer := range layers {
			workerDeltas[w][i] = layer.newDelta()
		}
	}
	aggregated := make([]*delta, len(layers))
	for i, layer := range layers {
		aggregated[i] = layer.newDelta()
	}
	updateRNGs := make([]*rand.Rand, len(layers))
	for i := range layers {
		var stream uint64
		switch {
		case m.mode == "dense":
			stream = 0x100 + uint64(i)
		case i < len(m.blocks):
			stream = 0x200 + uint64(i)
		default:
			// project と reversible の読み出し層は同じ乱数列を使う。
			stream = 0xABCDEF
		}
		updateRNGs[i] = rand.New(rand.NewPCG(seed, 0xA24BAED4963EE407+stream))
	}
	return &trainer{
		model: m, miniBatchSize: 1024, lr: 0.1, margin: 0.5, noiseScale: 0.5,
		workerRNGs:   workerRNGs,
		shuffleRNG:   rand.New(rand.NewPCG(seed, 0xD1B54A32D192ED03)),
		updateRNGs:   updateRNGs,
		workerDeltas: workerDeltas, aggregated: aggregated,
	}, nil
}

func (t *trainer) trainEpoch(xs bitsx.Matrices, labels []int) error {
	n := len(xs)
	batch := min(n, t.miniBatchSize)
	perm := t.shuffleRNG.Perm(n)
	for start := 0; start < n; start += batch {
		end := min(start+batch, n)
		idxs := perm[start:end]
		for _, ds := range t.workerDeltas {
			for _, d := range ds {
				d.clear()
			}
		}
		if err := parallel.For(len(idxs), len(t.workerRNGs), func(workerID, i int) error {
			idx := idxs[i]
			y, bws, err := t.model.forward(xs[idx], t.noiseScale, t.revNoiseScale, t.workerRNGs[workerID])
			if err != nil {
				return err
			}
			should, err := satisfiesUpdateCriterion(y, labels[idx], t.model.prototypes, t.margin)
			if err != nil || !should {
				return err
			}
			target := t.model.prototypes[labels[idx]]
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
			for i, d := range ds {
				t.aggregated[i].add(d)
			}
		}
		for _, d := range t.aggregated {
			d.sign()
		}
		for i, layer := range t.model.trainableLayers() {
			if t.model.mode != "dense" && i < len(t.model.blocks) && !t.model.blocks[i].learnable {
				continue
			}
			if err := layer.update(t.aggregated[i], t.lr, t.updateRNGs[i]); err != nil {
				return err
			}
		}
	}
	return nil
}

func splitTrainValidation(xs bitsx.Matrices, labels []int, ratio float64, rng *rand.Rand) (bitsx.Matrices, []int, bitsx.Matrices, []int, error) {
	valN := int(float64(len(xs)) * ratio)
	if valN <= 0 || valN >= len(xs) {
		return nil, nil, nil, nil, fmt.Errorf("valratio %g では分割できない", ratio)
	}
	perm := rng.Perm(len(xs))
	valXs := make(bitsx.Matrices, 0, valN)
	valLabels := make([]int, 0, valN)
	trainXs := make(bitsx.Matrices, 0, len(xs)-valN)
	trainLabels := make([]int, 0, len(xs)-valN)
	for i, idx := range perm {
		if i < valN {
			valXs = append(valXs, xs[idx])
			valLabels = append(valLabels, labels[idx])
		} else {
			trainXs = append(trainXs, xs[idx])
			trainLabels = append(trainLabels, labels[idx])
		}
	}
	return trainXs, trainLabels, valXs, valLabels, nil
}

func parameterCount(m *model) int {
	total := 0
	for _, layer := range m.trainableLayers() {
		total += layer.w.Rows()*layer.w.Cols() + len(layer.bias)
	}
	return total
}

func main() {
	var (
		mode        = flag.String("mode", "reversible", "dense | project | reversible")
		datasetName = flag.String("dataset", "mnist", "mnist | fashion")
		blocks      = flag.Int("blocks", 2, "可逆カップリングの段数")
		state       = flag.Int("state", 1024, "可逆状態の幅(128の倍数)")
		output      = flag.Int("output", 1024, "ETF出力幅")
		learnRev    = flag.Bool("learnrev", true, "可逆ブロック内部を学習する")
		useBias     = flag.Bool("bias", true, "学習する整数バイアスを使う")
		biasChoice  = flag.Float64("biaschoice", 0.1, "ニューロン更新でバイアス側を選ぶ確率")
		lr          = flag.Float64("lr", 0.1, "確率的学習率")
		margin      = flag.Float64("margin", 0.5, "BEP更新マージン")
		noise       = flag.Float64("noise", 0.5, "通常Denseの学習時ノイズ")
		revNoise    = flag.Float64("revnoise", 0, "可逆ブロック内部の学習時ノイズ")
		groupSize   = flag.Int("gsize", 4, "BEP GroupSize")
		gate        = flag.Float64("gate", 1, "BEP GateDropThresholdScale")
		batch       = flag.Int("batch", 1024, "ミニバッチサイズ")
		epochs      = flag.Int("epochs", 20, "エポック数")
		valRatio    = flag.Float64("valratio", 0.1, "検証分離率")
		seed        = flag.Uint64("seed", 1, "乱数シード")
		threads     = flag.Int("threads", 0, "ワーカー数(0=NumCPU)")
	)
	flag.Parse()
	workers := *threads
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	var ds dataset.Binary[int]
	var err error
	switch *datasetName {
	case "mnist":
		ds, err = dataset.LoadMNIST(nil)
	case "fashion":
		ds, err = dataset.LoadFashionMNIST(nil)
	default:
		log.Fatalf("dataset が不正: %s", *datasetName)
	}
	if err != nil {
		log.Fatal(err)
	}
	m, err := newModel(*mode, 784, *state, *output, *blocks, *learnRev, *useBias, float32(*biasChoice), *seed)
	if err != nil {
		log.Fatal(err)
	}
	for _, layer := range m.trainableLayers() {
		layer.groupSize = *groupSize
		layer.gateScale = float32(*gate)
	}
	trainer, err := newTrainer(m, workers, *seed)
	if err != nil {
		log.Fatal(err)
	}
	trainer.miniBatchSize = *batch
	trainer.lr = float32(*lr)
	trainer.margin = float32(*margin)
	trainer.noiseScale = float32(*noise)
	trainer.revNoiseScale = float32(*revNoise)
	splitRNG := rand.New(rand.NewPCG(*seed, 0xC2B2AE3D27D4EB4F))
	trainXs, trainLabels, valXs, valLabels, err := splitTrainValidation(ds.TrainInputs, ds.TrainLabels, *valRatio, splitRNG)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("dataset=%s mode=%s blocks=%d state=%d output=%d params=%d learnrev=%t bias=%t biaschoice=%g lr=%g margin=%g noise=%g revnoise=%g epochs=%d seed=%d train=%d val=%d test=%d\n",
		*datasetName, *mode, *blocks, *state, *output, parameterCount(m), *learnRev, *useBias, *biasChoice,
		*lr, *margin, *noise, *revNoise, *epochs, *seed, len(trainXs), len(valXs), len(ds.TestInputs))
	if *mode != "dense" {
		state0, err := m.backbone(ds.TestInputs[0])
		if err != nil {
			log.Fatal(err)
		}
		restored, err := m.inverseBackbone(state0)
		if err != nil {
			log.Fatal(err)
		}
		padded, err := padState(ds.TestInputs[0], m.stateCols)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("可逆性検査: restored=%t\n", restored.Equal(padded))
	}
	bestVal, testAtBest, bestEpoch := -1.0, 0.0, 0
	for epoch := 1; epoch <= *epochs; epoch++ {
		started := time.Now()
		if err := trainer.trainEpoch(trainXs, trainLabels); err != nil {
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
		if valAcc > bestVal {
			bestVal, testAtBest, bestEpoch = valAcc, testAcc, epoch
		}
		fmt.Printf("epoch %d: val %.4f test %.4f best %.4f@%d %.1fs\n", epoch, valAcc, testAcc, bestVal, bestEpoch, time.Since(started).Seconds())
	}
	fmt.Printf("BEST_VAL %.4f @epoch %d / TEST %.4f\n", bestVal, bestEpoch, testAtBest)
}
