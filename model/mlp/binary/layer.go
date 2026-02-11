package binary

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/sw965/omw/mathx"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	"github.com/sw965/omw/slicesx"
	"slices"
)

type H []int8

type SharedContext struct {
	GateThresholdScale float32
	NoiseStdScale      float64
	GroupSize          int
}

type Layer interface {
	Forward(bitsx.Matrix, *rand.Rand) (bitsx.Matrix, Backward, error)
	Predict(bitsx.Matrix) (bitsx.Matrix, error)
	NewZerosDeltas() Deltas
	Update(Deltas, float32, *rand.Rand) error
	setSharedContext(*SharedContext) error
}

type Backward func(bitsx.Matrix, Deltas) (bitsx.Matrix, error)
type Backwards []Backward

func (bs Backwards) Propagate(t bitsx.Matrix, seqDelta SeqDelta) (bitsx.Matrix, error) {
	var err error
	for layerI := len(bs) - 1; layerI >= 0; layerI-- {
		t, err = bs[layerI](t, seqDelta[layerI])
		if err != nil {
			return bitsx.Matrix{}, err
		}
	}
	return t, nil
}

type Dense struct {
	W             bitsx.Matrix
	wT            bitsx.Matrix
	H             []int8
	sharedContext *SharedContext

	gateThresholdBase int
	noiseStdBase      float64
}

func NewDense(wRows, wCols int, rng *rand.Rand) (Dense, error) {
	w, err := bitsx.NewRandMatrix(wRows, wCols, 0, rng)
	if err != nil {
		return Dense{}, fmt.Errorf("後でエラーメッセージを書く")
	}

	wt, err := w.Transpose()
	if err != nil {
		return Dense{}, fmt.Errorf("後でエラーメッセージを書く")
	}

	h := make(H, wRows*wCols)
	err = w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		wWord := w.Data[ctx.WordIndex]
		hWord := h[ctx.GlobalStart:ctx.GlobalEnd]
		ctx.ScanBits(func(i, col, colT int) error {
			wBit := wWord >> uint64(i) & 1
			if wBit == 1 {
				hWord[i] = math.MaxInt8 / 4
			} else {
				hWord[i] = math.MinInt8 / 4
			}
			return nil
		})
		return nil
	})

	noiseStdBase := math.Sqrt(float64(w.Cols))
	gateThresholdBase := int(noiseStdBase)

	return Dense{
		W:                 w,
		wT:                wt,
		H:                 h,
		gateThresholdBase: gateThresholdBase,
		noiseStdBase:      noiseStdBase,
	}, nil
}

func (d *Dense) GateThreshold() int {
	return int(d.sharedContext.GateThresholdScale * float32(d.gateThresholdBase))
}

func (d *Dense) NoiseStd() float64 {
	return d.sharedContext.NoiseStdScale * d.noiseStdBase
}

func (d *Dense) Forward(x bitsx.Matrix, rng *rand.Rand) (bitsx.Matrix, Backward, error) {
	u, err := x.Dot(d.W)
	if err != nil {
		return bitsx.Matrix{}, nil, err
	}

	maxZi := d.W.Cols
	minZi := -maxZi

	noiseStd := d.NoiseStd()
	isNoisy := noiseStd > 0.0

	yRows := x.Rows
	yCols := d.W.Rows
	z := make([]int, yRows*yCols)

	if isNoisy {
		for i, count := range u {
			zi := 2*count - maxZi
			noise, err := randx.NormalInt(minZi, maxZi, 0, noiseStd, rng)
			if err != nil {
				return bitsx.Matrix{}, nil, err
			}
			z[i] = zi + noise
		}
	} else {
		for i, count := range u {
			z[i] = 2*count - maxZi
		}
	}

	y, err := bitsx.NewSignMatrix(yRows, yCols, z)
	if err != nil {
		return bitsx.Matrix{}, nil, err
	}

	var backward Backward
	backward = func(t bitsx.Matrix, deltas Deltas) (bitsx.Matrix, error) {
		if t.Rows != y.Rows || t.Cols != y.Cols {
			return bitsx.Matrix{}, fmt.Errorf("後でエラーメッセージを書く")
		}

		keepGate, err := bitsx.NewZerosMatrix(yRows, yCols)
		if err != nil {
			return bitsx.Matrix{}, err
		}
		// 超えたらゲートを閉じる事を、表す為に、dropを命名に加える？
		gateThreshold := d.GateThreshold()
		err = t.ScanRowsWord(nil, func(tCtx bitsx.MatrixWordContext) error {
			// 64ビット毎に操作するための宣言
			zWord := z[tCtx.GlobalStart:tCtx.GlobalEnd]
			ascIdxs := slicesx.Argsort(zWord)
			cutoffIdx := len(ascIdxs) / d.sharedContext.GroupSize
			if cutoffIdx == 0 {
				cutoffIdx = 1
			}
			cutoffZi := zWord[ascIdxs[cutoffIdx-1]]
			tWord := t.Data[tCtx.WordIndex]
			var keepGateWord uint64

			// ScanBitsで上記の64ビット(Word)に対して操作する
			// 引数iに代入される値は、100ビットの場合、一週目は0～63、二週目は64～99
			tCtx.ScanBits(func(i, col, colT int) error {
				zi := zWord[i]
				if zi > cutoffZi {
					return nil
				}

				if mathx.Abs(zi) > gateThreshold {
					return nil
				}
				keepGateWord |= (1 << uint64(i))

				tBit := (tWord >> uint64(i)) & 1
				yBit := uint64(0)
				if zi >= 0 {
					yBit = 1
				}

				// 正解なら更新しない
				if tBit == yBit {
					return nil
				}

				outputCol := tCtx.ColStart + i
				deltaRow := deltas[0][outputCol*d.W.Cols : (outputCol+1)*d.W.Cols]

				// 64ビット毎にxを見て重みの勾配を求める
				x.ScanRowsWord([]int{tCtx.Row}, func(xCtx bitsx.MatrixWordContext) error {
					xWord := x.Data[xCtx.WordIndex]
					deltaWord := deltaRow[xCtx.ColStart:xCtx.ColEnd]
					for b := range deltaWord {
						xBit := (xWord >> uint(b)) & 1
						deltaWord[b] += int16(1 - 2*int(xBit^tBit))
					}
					return nil
				})
				return nil
			})

			keepGate.Data[tCtx.WordIndex] = keepGateWord
			return nil
		})

		if err != nil {
			return bitsx.Matrix{}, err
		}

		rawNextTT, err := d.wT.DotTernary(t, keepGate)
		if err != nil {
			return bitsx.Matrix{}, err
		}

		nextT, err := bitsx.NewZerosMatrix(yRows, d.W.Cols)
		if err != nil {
			return bitsx.Matrix{}, err
		}

		err = nextT.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
			var word uint64
			ctx.ScanBits(func(i, col, colT int) error {
				if rawNextTT[colT] >= 0 {
					word |= (1 << uint(i))
				}
				return nil
			})
			nextT.Data[ctx.WordIndex] = word
			return nil
		})

		if err != nil {
			return bitsx.Matrix{}, err
		}
		return nextT, nil
	}
	return y, backward, nil
}

func (d *Dense) Predict(x bitsx.Matrix) (bitsx.Matrix, error) {
	u, err := x.Dot(d.W)
	if err != nil {
		return bitsx.Matrix{}, err
	}

	maxZi := d.W.Cols
	z := make([]int, len(u))
	for i, count := range u {
		z[i] = 2*count - maxZi
	}

	yRows := x.Rows
	yCols := d.W.Rows
	y, err := bitsx.NewSignMatrix(yRows, yCols, z)
	if err != nil {
		return bitsx.Matrix{}, err
	}
	return y, nil
}

func (d *Dense) NewZerosDeltas() Deltas {
	return Deltas{make(Delta, d.W.Rows*d.W.Cols)}
}

func (d *Dense) Update(deltas Deltas, lr float32, rng *rand.Rand) error {
	if len(deltas) != 1 {
		return fmt.Errorf("後でエラーメッセージを書く")
	}

	delta := deltas[0]
	err := d.W.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		hWord := d.H[ctx.GlobalStart:ctx.GlobalEnd]
		deltaWord := delta[ctx.GlobalStart:ctx.GlobalEnd]
		var flip uint64
		ctx.ScanBits(func(i, col, colT int) error {
			if rng.Float32() > lr {
				return nil
			}

			old := hWord[i]
			// オーバーフロー対策に一旦intにする
			newVal := int(old) + int(deltaWord[i])
			clipped := int8(max(math.MinInt8, min(newVal, math.MaxInt8)))
			hWord[i] = clipped

			isOldPlus := old >= 0
			isNewPlus := clipped >= 0
			if isOldPlus != isNewPlus {
				flip |= (1 << uint64(i))
				err := d.wT.Toggle(col, ctx.Row)
				if err != nil {
					return err
				}
			}
			return nil
		})
		d.W.Data[ctx.WordIndex] ^= flip
		return nil
	})
	return err
}

func (d *Dense) setSharedContext(ctx *SharedContext) error {
	d.sharedContext = ctx
	return nil
}

type Sequence []Layer

func NewDenseLayers(dims []int, rng *rand.Rand) (Sequence, error) {
	numLayers := len(dims)-1
	if numLayers < 1 {
		return nil, fmt.Errorf("layerSizes must have at least 2 elements (input and output dimensions)")
	}
	seq := make(Sequence, 0, numLayers)
	for i := range numLayers {
		wRows := dims[i+1]
		wCols := dims[i]

		denseLayer, err := NewDense(wRows, wCols, rng)
		if err != nil {
			return nil, fmt.Errorf("failed to create dense layer at index %d: %w", i, err)
		}
		seq = append(seq, &denseLayer)
	}
	return seq, nil
}

func (s Sequence) Forwards(x bitsx.Matrix, rng *rand.Rand) (bitsx.Matrix, Backwards, error) {
	var backward Backward
	var err error
	backwards := make(Backwards, len(s))
	for i, layer := range s {
		x, backward, err = layer.Forward(x, rng)
		if err != nil {
			return bitsx.Matrix{}, nil, err
		}
		backwards[i] = backward
	}
	y := x
	return y, backwards, nil
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

func (s Sequence) PredictSoftmax(x bitsx.Matrix, prototypes bitsx.Matrices) ([]float32, error) {
	logits, err := s.PredictLogits(x, prototypes)
	if err != nil {
		return nil, err
	}

	if len(logits) == 0 {
		return nil, fmt.Errorf("logits is empty")
	}

	maxLogit := slices.Max(logits)
	exps := make([]float64, len(logits))
	var sumExp float64
	for i, l := range logits {
		exps[i] = math.Exp(float64(l - maxLogit))
		sumExp += exps[i]
	}

	y := make([]float32, len(logits))
	for i, exp := range exps {
		y[i] = float32(exp / sumExp)
	}
	return y, nil
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

func (s Sequence) Update(seqDelta SeqDelta, lr float32, rngs []*rand.Rand) error {
	if len(s) != len(seqDelta) {
		return fmt.Errorf("sequence and delta length mismatch: %d != %d", len(s), len(seqDelta))
	}

	p := len(rngs)
	numLayers := len(s)
	if p > numLayers {
		p = numLayers
	}

	err := parallel.For(numLayers, p, func(workerId, idx int) error {
		layer := s[idx]
		layerDelta := seqDelta[idx]
		rng := rngs[workerId]
		return layer.Update(layerDelta, lr, rng)
	})
	return err
}

func (s Sequence) SetSharedContext(ctx *SharedContext) error {
	for i := range s {
		err := s[i].setSharedContext(ctx)
		if err != nil {
			return err
		}
	}
	return nil
}