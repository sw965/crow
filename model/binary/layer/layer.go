package layer

import (
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/sw965/omw/mathx"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
)

type Delta []int16

type TrainingContext struct {
	GateThresholdScale     float32
    NoiseStdScale          float64
    GroupSize              int
    TriggerCount           int64
}

type Interface interface {
	Forward(bitsx.Matrix) (bitsx.Matrix, error)
	Backward(bitsx.Matrix) (bitsx.Matrix, error)
	Predict(bitsx.Matrix) (bitsx.Matrix, error)
	Delta() Delta
	ClearDelta()
}

type Dense struct {
	W   bitsx.Matrix
	WT  bitsx.Matrix
	Rng *rand.Rand

	TrainingContext *TrainingContext

	x     bitsx.Matrix
	zs    []int
	delta Delta

	gateThresholdBase int
	noiseStdBase      float64
}

func NewDense(w, wt bitsx.Matrix, ctx *TrainingContext) (*Dense, error) {
	if w.Rows != wt.Cols {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	if w.Cols != wt.Rows {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	n := w.Rows * w.Cols
	noiseStdBase := math.Sqrt(float64(w.Cols))
	gateThresholdBase := int(noiseStdBase)
	return &Dense{
		W:             w,
		WT:            wt,
		TrainingContext: ctx,
		zs:            make([]int, n),
		delta:         make([]int16, n),
		gateThresholdBase: gateThresholdBase,
		noiseStdBase:      noiseStdBase,
	}, nil
}

func (d *Dense) GateThreshold() int {
	return int(d.TrainingContext.GateThresholdScale * float32(d.gateThresholdBase))
}

func (d *Dense) NoiseStd() float64 {
	return d.TrainingContext.NoiseStdScale * d.noiseStdBase
}

func (d *Dense) Forward(x bitsx.Matrix) (bitsx.Matrix, error) {
	u, err := x.Dot(d.W)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	maxZ := d.W.Cols
	minZ := -maxZ

	noiseStd := d.NoiseStd()
	isNoisy := noiseStd > 0.0

	if isNoisy {
		for i, count := range u {
			z := 2*count - maxZ
			noise, err := randx.NormalInt(minZ, maxZ, 0, noiseStd, d.Rng)
			if err != nil {
				return bitsx.Matrix{}, err
			}
			d.zs[i] = z + noise
		}
	} else {
		for i, count := range u {
			d.zs[i] = 2*count - maxZ
		}
	}

	yRows := x.Rows
	yCols := d.W.Rows
	sign, err := bitsx.NewSignMatrix(yRows, yCols, d.zs)
	if err != nil {
		return bitsx.Matrix{}, err
	}
	d.x = x
	return sign, nil
}

func (d *Dense) Backward(target bitsx.Matrix) (bitsx.Matrix, error) {
	yRows, yCols := target.Rows, target.Cols
	wCols := d.W.Cols

	if yRows != d.x.Rows || yCols != d.W.Rows {
		return bitsx.Matrix{}, fmt.Errorf("target shape mismatch")
	}

	keepGate, err := bitsx.NewZerosMatrix(yRows, yCols)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	gateThreshold := d.GateThreshold()

	// wRows := d.W.Rows
	err = target.ScanRowsWord(nil, func(tCtx bitsx.MatrixWordContext) error {
		// 64ビット毎に操作するための宣言
		zsWord := d.zs[tCtx.GlobalStart:tCtx.GlobalEnd]
		ascIdxs := slicesx.Argsort(zsWord)
		cutoffZIdx := len(ascIdxs) / d.TrainingContext.GroupSize
		if cutoffZIdx == 0 {
			cutoffZIdx = 1
		}
		cutoffZ := zsWord[ascIdxs[cutoffZIdx]]
		tWord := target.Data[tCtx.WordIndex]
		var keepGateWord uint64

		// ScanBitsで上記の64ビット(Word)に対して操作する
		// 引数iに代入される値は、100ビットの場合、一週目は0～63、二週目は64～99
		tCtx.ScanBits(func(i, col, colT int) {
			z := zsWord[i]
			if z > cutoffZ {
				return
			}

			if mathx.Abs(z) > gateThreshold {
				return
			}
			keepGateWord |= (1 << uint64(i))

			tBit := (tWord >> uint64(i)) & 1
			yBit := uint64(0)
			if z >= 0 {
				yBit = 1
			}

			// 正解なら更新しない
			if tBit == yBit {
				return
			}

			outputCol := tCtx.ColStart + i
			deltaRow := d.delta[outputCol*wCols : (outputCol+1)*wCols]

			// 64ビット毎にxを見て重みの勾配を求める
			d.x.ScanRowsWord([]int{tCtx.Row}, func(xCtx bitsx.MatrixWordContext) error {
				xWord := d.x.Data[xCtx.WordIndex]
				deltaWord := deltaRow[xCtx.ColStart:xCtx.ColEnd]
				for b := range deltaWord {
					xBit := (xWord >> uint(b)) & 1
					deltaWord[b] += int16(1 - 2*int(xBit^tBit))
				}
				return nil
			})
		})

		keepGate.Data[tCtx.WordIndex] = keepGateWord
		return nil
	})

	if err != nil {
		return bitsx.Matrix{}, err
	}

	rawNextTargetT, err := d.WT.DotTernary(target, keepGate)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	nextTarget, err := bitsx.NewZerosMatrix(yRows, wCols)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	err = nextTarget.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		var word uint64
		ctx.ScanBits(func(i, col, colT int) {
			if rawNextTargetT[colT] >= 0 {
				word |= (1 << uint(i))
			}
		})
		nextTarget.Data[ctx.WordIndex] = word
		return nil
	})

	if err != nil {
		return bitsx.Matrix{}, err
	}
	return nextTarget, nil
}

func (d *Dense) Predict(x bitsx.Matrix) (bitsx.Matrix, error) {
	u, err := x.Dot(d.W)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	maxZ := d.W.Cols
	zs := make([]int, len(u))
	for i, count := range u {
		zs[i] = 2*count - maxZ
	}

	yRows := x.Rows
	yCols := d.W.Rows
	sign, err := bitsx.NewSignMatrix(yRows, yCols, zs)
	if err != nil {
		return bitsx.Matrix{}, err
	}
	return sign, nil
}

func (d *Dense) Delta() Delta {
	return d.delta
}

func (d *Dense) ClearDelta() {
	clear(d.delta)
}

type Sequence []Interface

func (s Sequence) Forwards(x bitsx.Matrix) (bitsx.Matrix, error) {
	var err error
	for _, layer := range s {
		x, err = layer.Forward(x)
		if err != nil {
			return bitsx.Matrix{}, err
		}
	}
	y := x
	return y, nil
}

func (s Sequence) Backwards(target bitsx.Matrix) (bitsx.Matrix, error) {
	var err error
	for layerI := len(s) - 1; layerI >= 0; layerI-- {
		target, err = s[layerI].Backward(target)
		if err != nil {
			return bitsx.Matrix{}, err
		}
	}
	return target, nil
}

func (s Sequence) Predict(x bitsx.Matrix) (bitsx.Matrix, error) {
	var err error
	for _, layer := range s {
		x, err = layer.Predict(x)
		if err != nil {
			return bitsx.Matrix{}, err
		}
	}
	return x, nil
}

func (s Sequence) PredictLogits(x bitsx.Matrix, prototypes bitsx.Matrices) ([]int, error) {
	y, err := s.Predict(x)
	if err != nil {
		return nil, err
	}

	n := len(prototypes)
	logits := make([]int, n)
	maxMatch := y.Rows * y.Cols

	for i, proto := range prototypes {
		if y.Rows != proto.Rows || y.Cols != proto.Cols {
			return nil, fmt.Errorf("後でエラーメッセージを書く")
		}
		mismatch, err := y.HammingDistance(proto)
		if err != nil {
			return nil, err
		}
		logits[i] = maxMatch - mismatch
	}
	return logits, nil
}

func (s Sequence) Accuracy(xs bitsx.Matrices, labels []int, prototypes bitsx.Matrices, p int) (float32, error) {
	n := len(xs)
	if n != len(labels) {
		return 0.0, fmt.Errorf("length mismatch: xs %d != labels %d", n, len(labels))
	}
	correctCounts := make([]int, p)

	err := parallel.For(n, p, func(workerId, idx int) error {
		x := xs[idx]
		label := labels[idx]

		logits, err := s.PredictLogits(x, prototypes)
		if err != nil {
			return err
		}

		if len(logits) == 0 {
			return fmt.Errorf("logits is empty")
		}

		yMaxIdx := slicesx.Argsort(logits)[len(logits)-1]
		if yMaxIdx == label {
			correctCounts[workerId]++
		}
		return nil
	})

	if err != nil {
		return 0.0, err
	}

	totalCorrect := 0
	for _, c := range correctCounts {
		totalCorrect += c
	}
	return float32(totalCorrect) / float32(n), nil
}

func (s Sequence) ClearDelta() {
	for _, layer := range s {
		layer.ClearDelta()
	}
}

type Sequences []Sequence

func (ss Sequences) aggregateDeltas() ([]Delta, error) {
	numLayers := len(ss[0])
	signDeltas := make([]Delta, numLayers)
	for layerI := range numLayers {
		worker0Delta := ss[0][layerI].Delta()
		worker0DeltaN := len(worker0Delta)
		sum := make([]int, worker0DeltaN)

		for _, seq := range ss {
			delta := seq[layerI].Delta()
			if len(delta) != worker0DeltaN {
				return nil, fmt.Errorf("後でエラーメッセージを書く")
			}

			for i, v := range delta {
				sum[i] += int(v)
			}
		}

		sign := make(Delta, worker0DeltaN)
		for i, v := range sum {
			sign[i] = int16(cmp.Compare(v, 0))
		}
		signDeltas[layerI] = sign
	}
	return signDeltas, nil
}

func (ss Sequences) ComputeSignDeltas(xs, ts bitsx.Matrices) ([]Delta, error) {
	n := len(xs)
	if n != len(ts) {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	// バッチサイズがmath.MaxInt16より大きいと、オーバーフローが起きる可能性があるため、ガードする
	// ※ ワーカー毎にDeltaを切り分けるが、合計して集約するステップがある事に注意
	if n > math.MaxInt16 {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}
	
	for _, seq := range ss {
		seq.ClearDelta()
	}
	p := len(ss)

	err := parallel.For(n, p, func(workerId, idx int) error {
		x := xs[idx]
		t := ts[idx]
		seq := ss[workerId]

		_, err := seq.Forwards(x)
		if err != nil {
			return err
		}

		_, err = seq.Backwards(t)
		if err != nil {
			return err
		}
		return nil
	})

	if err != nil {
		return nil, err
	}
	return ss.aggregateDeltas()
}

func (ss Sequences) ComputeSignDeltasForClassification(xs bitsx.Matrices, labels []int, prototypes bitsx.Matrices, ctx *TrainingContext) ([]Delta, error) {
	n := len(xs)
	if n != len(labels) {
		return nil, fmt.Errorf("length mismatch: xs %d != labels %d", n, len(labels))
	}

	for _, seq := range ss {
		seq.ClearDelta()
	}
	p := len(ss)

	err := parallel.For(n, p, func(workerId, idx int) error {
		x := xs[idx]
		label := labels[idx]
		seq := ss[workerId]

		y, err := seq.Forwards(x)
		if err != nil {
			return err
		}

		needsUpdate, err := NeedsClassificationUpdate(y, label, prototypes)
		if err != nil {
			return err
		}

		if !needsUpdate {
			return nil
		}

		t := prototypes[label]
		_, err = seq.Backwards(t)
		if err != nil {
			return err
		}

		ctx.TriggerCount += 1
		return nil
	})

	if err != nil {
		return nil, err
	}
	return ss.aggregateDeltas()
}

func NeedsClassificationUpdate(y bitsx.Matrix, label int, prototypes bitsx.Matrices) (bool, error) {
	t := prototypes[label]
	yMismatch, err := y.HammingDistance(t)
	if err != nil {
		return false, err
	}

	marginBits := (y.Rows*y.Cols) / 4
	for i, proto := range prototypes {
		if i == label {
			continue
		}

		mismatch, err := y.HammingDistance(proto)
		if err != nil {
			return false, err
		}

		if (mismatch - yMismatch) < marginBits {
    		return true, nil
		}
	}
	return false, nil
}

// NeedsRegressionUpdate