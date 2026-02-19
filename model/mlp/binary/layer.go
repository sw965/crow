package binary

import (
	"encoding/gob"
	"fmt"
	"github.com/sw965/omw/encoding/gobx"
	"github.com/sw965/omw/mathx"
	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/parallel"
	//"github.com/sw965/omw/slicesx"
	"cmp"
	"math"
	"math/rand/v2"
	"slices"
)

type H []int8

type SharedContext struct {
	GateDropThresholdScale float32
	NoiseStdScale          float32
	GroupSize              int
}

type Layer interface {
	Forward(*bitsx.Matrix, *rand.Rand) (*bitsx.Matrix, Backward, error)
	Predict(*bitsx.Matrix) (*bitsx.Matrix, error)
	NewZerosDeltas() Deltas
	OutputShape(xRows, xCols int) (int, int, error)
	Update(Deltas, float32, *rand.Rand) error
	setSharedContext(*SharedContext) error
}

type Backward func(*bitsx.Matrix, Deltas) (*bitsx.Matrix, error)
type Backwards []Backward

func (bs Backwards) Propagate(t *bitsx.Matrix, seqDelta SeqDelta) (*bitsx.Matrix, error) {
	var err error
	for layerI := len(bs) - 1; layerI >= 0; layerI-- {
		t, err = bs[layerI](t, seqDelta[layerI])
		if err != nil {
			return nil, err
		}
	}
	return t, nil
}

type Dense struct {
	W  *bitsx.Matrix
	WT *bitsx.Matrix
	H  []int8

	GateDropThresholdBase int
	NoiseStdBase          float32
	sharedContext         *SharedContext
}

func NewDense(wRows, wCols int, rng *rand.Rand) (*Dense, error) {
	w, err := bitsx.NewRandMatrix(wRows, wCols, 0, rng)
	if err != nil {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	wt, err := w.Transpose()
	if err != nil {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
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

	// mathx.Sqrtにする？
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
	return int(d.sharedContext.GateDropThresholdScale * float32(d.GateDropThresholdBase))
}

func (d *Dense) NoiseStd() float32 {
	return d.sharedContext.NoiseStdScale * d.NoiseStdBase
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
				absZi := mathx.Abs(zi)

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
					wordMismatches = append(wordMismatches, wordMismatch{absZi: absZi, tBit: tBit, col:col})
				}
				return nil
			})

			keepGate.Data[tCtx.WordIndex] = keepGateWord

			slices.SortFunc(wordMismatches, func(a, b wordMismatch) int {
				return cmp.Compare(a.absZi, b.absZi)
			})

			updateK := len(zWord) / d.sharedContext.GroupSize
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
		return 0, 0, fmt.Errorf("input cols %d does not match layer input dim %d", xCols, d.W.Cols)
	}
	return xRows, d.W.Rows, nil
}

func (d *Dense) Update(deltas Deltas, lr float32, rng *rand.Rand) error {
	if len(deltas) != 1 {
		return fmt.Errorf("後でエラーメッセージを書く")
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

			isOldPlus := old >= 0
			isNewPlus := clipped >= 0
			if isOldPlus != isNewPlus {
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

func (d *Dense) setSharedContext(ctx *SharedContext) error {
	d.sharedContext = ctx
	return nil
}

type Sequence []Layer

func LoadSequence(path string) (Sequence, error) {
	return gobx.Load[Sequence](path)
}

func (s Sequence) Forwards(x *bitsx.Matrix, rng *rand.Rand) (*bitsx.Matrix, Backwards, error) {
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
			return 0, 0, fmt.Errorf("後でエラーメッセージを書く")
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

func init() {
	gob.Register(&Dense{})
}
