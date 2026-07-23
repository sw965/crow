package binary

import (
	"cmp"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/sw965/omw/encoding/gobx"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
)

type H []int8

// Hの初期絶対値。大きくすると重み反転までに必要な更新回数が増え、序盤の学習が沈黙する。
// bep_report 実験3: ±31では3〜4エポック完全沈黙、±4なら1エポック目から立ち上がる。
const hInitAbs = 4

type SharedHyperparameters struct {
	GateDropThresholdScale float32
	NoiseStdScale          float32
	GroupSize              int
}

func NewSharedHyperparameters() SharedHyperparameters {
	return SharedHyperparameters{
		GateDropThresholdScale: 1.0,
		NoiseStdScale:          0.5,
		GroupSize:              4,
	}
}

type Layer interface {
	Forward(*bitsx.Matrix, *rand.Rand) (*bitsx.Matrix, Backward, error)
	Predict(*bitsx.Matrix) (*bitsx.Matrix, error)
	NewZerosDeltas() Deltas
	OutputShape(xRows, xCols int) (int, int, error)
	Update(Deltas, float32, *rand.Rand) error
	setSharedHyperparameters(*SharedHyperparameters) error
}

type Backward func(*bitsx.Matrix, Deltas) (*bitsx.Matrix, error)
type Backwards []Backward

func (bs Backwards) Propagate(t *bitsx.Matrix, seqDelta SeqDelta) (*bitsx.Matrix, error) {
	var err error
	for layerIdx := len(bs) - 1; layerIdx >= 0; layerIdx-- {
		t, err = bs[layerIdx](t, seqDelta[layerIdx])
		if err != nil {
			return nil, err
		}
	}
	return t, nil
}

type Dense struct {
	W  *bitsx.Matrix
	WT *bitsx.Matrix
	H  H

	GateDropThresholdBase int
	NoiseStdBase          float32
	sharedHyperparameters *SharedHyperparameters
}

func NewDense(wRows, wCols int, rng *rand.Rand) (*Dense, error) {
	w, err := bitsx.NewRandMatrix(wRows, wCols, 0, rng)
	if err != nil {
		return nil, fmt.Errorf("重み行列の生成に失敗: %w", err)
	}

	wt, err := w.Transpose()
	if err != nil {
		return nil, fmt.Errorf("重み行列の転置に失敗: %w", err)
	}

	h := make(H, wRows*wCols)
	err = w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		wWord := w.Data[ctx.WordIndex]
		hWord := h[ctx.GlobalStart:ctx.GlobalEnd]
		ctx.ScanBits(func(i, col, colT int) error {
			wBit := wWord >> uint64(i) & 1
			if wBit == 1 {
				hWord[i] = hInitAbs
			} else {
				hWord[i] = -hInitAbs
			}
			return nil
		})
		return nil
	})

	noiseStdBase := float32(math.Sqrt(float64(w.Cols)))
	gateDropThresholdBase := int(noiseStdBase)

	return &Dense{
		W:                     w,
		WT:                    wt,
		H:                     h,
		GateDropThresholdBase: gateDropThresholdBase,
		NoiseStdBase:          noiseStdBase,
	}, nil
}

func (d *Dense) GateDropThreshold() int {
	return int(d.sharedHyperparameters.GateDropThresholdScale * float32(d.GateDropThresholdBase))
}

func (d *Dense) NoiseStd() float32 {
	return d.sharedHyperparameters.NoiseStdScale * d.NoiseStdBase
}

func (d *Dense) Forward(x *bitsx.Matrix, rng *rand.Rand) (*bitsx.Matrix, Backward, error) {
	u, err := x.Dot(d.W)
	if err != nil {
		return nil, nil, err
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
				return nil, nil, err
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
		return nil, nil, err
	}

	var backward Backward
	backward = func(t *bitsx.Matrix, deltas Deltas) (*bitsx.Matrix, error) {
		if err := t.ValidateSameShape(y); err != nil {
			return nil, err
		}

		keepGate, err := bitsx.NewZerosMatrix(yRows, yCols)
		if err != nil {
			return nil, err
		}
		gateDropThreshold := d.GateDropThreshold()

		wordSize := 64
		err = t.ScanRowsWord(nil, func(tCtx bitsx.MatrixWordContext) error {
			// 64ビット毎に操作するための宣言
			zWord := z[tCtx.GlobalStart:tCtx.GlobalEnd]
			type wordMismatch struct {
				absZi int
				tBit  uint64
				col   int
			}
			wordMismatches := make([]wordMismatch, 0, wordSize)

			tWord := t.Data[tCtx.WordIndex]
			var keepGateWord uint64

			// ScanBitsで上記の64ビット(Word)に対して操作する
			// 引数iに代入される値は、100ビットの場合、一週目は0～63、二週目は64～99
			tCtx.ScanBits(func(i, col, colT int) error {
				zi := zWord[i]
				absZi := int(math.Abs(float64(zi)))

				if absZi <= gateDropThreshold {
					keepGateWord |= (1 << uint64(i))
				}

				tBit := (tWord >> uint64(i)) & 1
				yBit := uint64(0)
				if zi >= 0 {
					yBit = 1
				}

				// 不正解なら更新対象
				if tBit != yBit {
					wordMismatches = append(wordMismatches, wordMismatch{absZi: absZi, tBit: tBit, col: col})
				}
				return nil
			})

			keepGate.Data[tCtx.WordIndex] = keepGateWord

			slices.SortFunc(wordMismatches, func(a, b wordMismatch) int {
				return cmp.Compare(a.absZi, b.absZi)
			})

			updateK := len(zWord) / d.sharedHyperparameters.GroupSize
			if updateK == 0 {
				updateK = 1
			}

			if updateK > len(wordMismatches) {
				updateK = len(wordMismatches)
			}

			for _, mismatch := range wordMismatches[:updateK] {
				tBit := mismatch.tBit
				col := mismatch.col
				deltaRow := deltas[0][col*d.W.Cols : (col+1)*d.W.Cols]

				x.ScanRowsWord([]int{tCtx.Row}, func(xCtx bitsx.MatrixWordContext) error {
					xWord := x.Data[xCtx.WordIndex]
					deltaWord := deltaRow[xCtx.ColStart:xCtx.ColEnd]
					for b := range deltaWord {
						xBit := (xWord >> uint(b)) & 1
						deltaWord[b] += int16(1 - 2*int(xBit^tBit))
					}
					return nil
				})
			}
			return nil
		})

		if err != nil {
			return nil, err
		}

		rawNextTT, err := d.WT.DotTernary(t, keepGate)
		if err != nil {
			return nil, err
		}

		nextT, err := bitsx.NewZerosMatrix(yRows, d.W.Cols)
		if err != nil {
			return nil, err
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
			return nil, err
		}
		return nextT, nil
	}
	return y, backward, nil
}

func (d *Dense) Predict(x *bitsx.Matrix) (*bitsx.Matrix, error) {
	u, err := x.Dot(d.W)
	if err != nil {
		return nil, err
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
		return nil, err
	}
	return y, nil
}

func (d *Dense) NewZerosDeltas() Deltas {
	n := d.W.Rows * d.W.Cols
	return Deltas{make(Delta, n)}
}

func (d *Dense) OutputShape(xRows, xCols int) (int, int, error) {
	if xCols != d.W.Cols {
		return 0, 0, fmt.Errorf("入力の列数が不一致: xCols = %d, W.Cols = %d", xCols, d.W.Cols)
	}
	return xRows, d.W.Rows, nil
}

func (d *Dense) Update(deltas Deltas, lr float32, rng *rand.Rand) error {
	if len(deltas) != 1 {
		return fmt.Errorf("Deltasの数が不正: len(deltas) = %d: Dense層は1つのDeltaを持つべき", len(deltas))
	}

	delta := deltas[0]
	err := d.W.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		hWord := d.H[ctx.GlobalStart:ctx.GlobalEnd]
		deltaWord := delta[ctx.GlobalStart:ctx.GlobalEnd]
		var flips uint64
		ctx.ScanBits(func(i, col, colT int) error {
			if rng.Float32() > lr {
				return nil
			}

			old := hWord[i]
			// オーバーフロー対策に一旦intにする
			newVal := int(old) + int(deltaWord[i])
			clipped := int8(max(math.MinInt8, min(newVal, math.MaxInt8)))
			hWord[i] = clipped

			oldIsNonNegative := old >= 0
			newIsNonNegative := clipped >= 0
			if oldIsNonNegative != newIsNonNegative {
				flips |= (1 << uint64(i))
				err := d.WT.Toggle(col, ctx.Row)
				if err != nil {
					return err
				}
			}
			return nil
		})
		d.W.Data[ctx.WordIndex] ^= flips
		return nil
	})
	return err
}

func (d *Dense) setSharedHyperparameters(ctx *SharedHyperparameters) error {
	d.sharedHyperparameters = ctx
	return nil
}

type Sequence []Layer

func LoadSequence(path string) (Sequence, error) {
	return gobx.Load[Sequence](path)
}

func (s Sequence) Forward(x *bitsx.Matrix, rng *rand.Rand) (*bitsx.Matrix, Backwards, error) {
	var backward Backward
	var err error
	backwards := make(Backwards, len(s))
	for i, layer := range s {
		x, backward, err = layer.Forward(x, rng)
		if err != nil {
			return nil, nil, err
		}
		backwards[i] = backward
	}
	y := x
	return y, backwards, nil
}

func (s Sequence) Predict(x *bitsx.Matrix) (*bitsx.Matrix, error) {
	var err error
	for _, layer := range s {
		x, err = layer.Predict(x)
		if err != nil {
			return nil, err
		}
	}
	return x, nil
}

func (s Sequence) OutputShape(xRows, xCols int) (int, int, error) {
	var err error
	for i, layer := range s {
		xRows, xCols, err = layer.OutputShape(xRows, xCols)
		if err != nil {
			return 0, 0, fmt.Errorf("layer %d: %w", i, err)
		}
		if xRows <= 0 || xCols <= 0 {
			return 0, 0, fmt.Errorf("layer %d: 出力形状が不正: rows = %d, cols = %d: どちらも正であるべき", i, xRows, xCols)
		}
	}
	yRows, yCols := xRows, xCols
	return yRows, yCols, nil
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

	err := parallel.For(numLayers, p, func(workerID, idx int) error {
		layer := s[idx]
		layerDelta := seqDelta[idx]
		rng := rngs[workerID]
		return layer.Update(layerDelta, lr, rng)
	})
	return err
}

func (s Sequence) SetSharedHyperparameters(ctx *SharedHyperparameters) error {
	for i := range s {
		err := s[i].setSharedHyperparameters(ctx)
		if err != nil {
			return err
		}
	}
	return nil
}

func init() {
	gob.Register(&Dense{})
}
