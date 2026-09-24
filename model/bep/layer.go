package bep

import (
	"cmp"
	"encoding/gob"
	"fmt"
	"math"
	"math/big"
	"math/bits"
	"math/rand/v2"
	"slices"

	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/parallel"
)

type H []int8

// Hの初期絶対値。大きくすると重み反転までに必要な更新回数が増え、序盤の学習が沈黙する。
// bep_report 実験3: ±31では3〜4エポック完全沈黙、±4なら1エポック目から立ち上がる。
const hInitAbs = 4

const defaultGroupSize = 4

type Layer interface {
	Forward(x *bitsx.Matrix, rng *rand.Rand) (*bitsx.Matrix, Backward, error)
	Predict(x *bitsx.Matrix) (*bitsx.Matrix, error)
	NewZerosDeltas() Deltas
	OutputShape(xRows, xCols int) (int, int, error)
	Update(deltas Deltas, lrHalfPow int, rng *rand.Rand) error
}

type Backward func(*bitsx.Matrix, Deltas) (*bitsx.Matrix, error)
type Backwards []Backward

func (bs Backwards) Propagate(t *bitsx.Matrix, seqDelta SeqDelta) (*bitsx.Matrix, error) {
	var err error
	for layerIdx := range slices.Backward(bs) {
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

	GroupSize   int
	MaxAbsNoise int
}

func isqrt(n int) int {
	r := 0
	for (r+1)*(r+1) <= n {
		r++
	}
	return r
}

// 倍率 num/denom は、ノイズの標準偏差を z の自然なばらつき √fanIn の何倍にするかを表す。
// 一様分布 [-w, w] の標準偏差は約 w/√3 なので、w = (num/denom) × √3 × √fanIn とする(√3 ≈ 1732/1000)。
// 極端な num で掛け算が桁あふれしないよう big.Int で計算する。
func maxAbsNoiseForScale(fanIn, num, denom int) (int, error) {
	if num < 0 || denom <= 0 {
		return 0, fmt.Errorf("ノイズの倍率が不正: num = %d, denom = %d: num >= 0 かつ denom > 0 であるべき", num, denom)
	}
	w := new(big.Int).Mul(big.NewInt(int64(isqrt(fanIn))*1732), big.NewInt(int64(num)))
	w.Quo(w, big.NewInt(1000))
	w.Quo(w, big.NewInt(int64(denom)))
	if !w.IsInt64() || w.Int64() > int64(fanIn) {
		return 0, fmt.Errorf("ノイズの倍率が大きすぎる: num/denom = %d/%d: MaxAbsNoise が入力数 %d を超える", num, denom, fanIn)
	}
	return int(w.Int64()), nil
}

func NewDense(wRows, wCols int, rng *rand.Rand) (*Dense, error) {
	w, err := bitsx.NewRandMatrix(wRows, wCols, rng)
	if err != nil {
		return nil, fmt.Errorf("重み行列の生成に失敗: %w", err)
	}

	wt, err := w.Transpose()
	if err != nil {
		return nil, fmt.Errorf("重み行列の転置に失敗: %w", err)
	}

	h := make(H, wRows*wCols)
	if err = w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		wWord, err := w.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		hWord := h[ctx.GlobalStart:ctx.GlobalEnd]
		err = ctx.ScanBits(func(i, col, colT int) error {
			wBit := wWord >> uint64(i) & 1
			if wBit == 1 {
				hWord[i] = hInitAbs
			} else {
				hWord[i] = -hInitAbs
			}
			return nil
		})
		return err
	}); err != nil {
		return nil, err
	}

	// 倍率 1/2 は旧実装のガウスノイズの既定値で、1/2・3/4・1 の実測で最も精度が良かった
	maxAbsNoise, err := maxAbsNoiseForScale(w.Cols(), 1, 2)
	if err != nil {
		return nil, err
	}

	return &Dense{
		W:           w,
		WT:          wt,
		H:           h,
		GroupSize:   defaultGroupSize,
		MaxAbsNoise: maxAbsNoise,
	}, nil
}

func (d *Dense) SetNoiseScale(num, denom int) error {
	maxAbsNoise, err := maxAbsNoiseForScale(d.W.Cols(), num, denom)
	if err != nil {
		return err
	}
	d.MaxAbsNoise = maxAbsNoise
	return nil
}

func (d *Dense) preActivation(x *bitsx.Matrix) ([]int, error) {
	u, err := x.Dot(d.W)
	if err != nil {
		return nil, err
	}

	maxZi := d.W.Cols()
	z := make([]int, len(u))
	for i, count := range u {
		z[i] = 2*count - maxZi
	}
	return z, nil
}

func (d *Dense) Forward(x *bitsx.Matrix, rng *rand.Rand) (*bitsx.Matrix, Backward, error) {
	z, err := d.preActivation(x)
	if err != nil {
		return nil, nil, err
	}

	maxAbsNoise := d.MaxAbsNoise
	isNoisy := maxAbsNoise > 0
	if isNoisy {
		for i := range z {
			z[i] += rng.IntN(2*maxAbsNoise+1) - maxAbsNoise
		}
	}

	y, err := bitsx.NewSignMatrix(x.Rows(), d.W.Rows(), z)
	if err != nil {
		return nil, nil, err
	}

	backward := func(t *bitsx.Matrix, deltas Deltas) (*bitsx.Matrix, error) {
		if err := t.ValidateSameShape(y); err != nil {
			return nil, err
		}
		if err := d.accumulateDelta(x, z, t, deltas[0]); err != nil {
			return nil, err
		}
		return d.inputTarget(t)
	}
	return y, backward, nil
}

func (d *Dense) accumulateDelta(x *bitsx.Matrix, z []int, t *bitsx.Matrix, delta Delta) error {
	return t.ScanRowsWord(nil, func(tCtx bitsx.MatrixWordContext) error {
		// 64ビット毎に操作するための宣言
		zWord := z[tCtx.GlobalStart:tCtx.GlobalEnd]
		type wordMismatch struct {
			absZi int
			tBit  uint64
			col   int
		}
		wordMismatches := make([]wordMismatch, 0, 64)

		tWord, err := t.Word(tCtx.WordIndex)
		if err != nil {
			return err
		}

		// ScanBitsで上記の64ビット(Word)に対して操作する
		// 引数iに代入される値は、100ビットの場合、一週目は0～63、二週目は64～99
		err = tCtx.ScanBits(func(i, col, colT int) error {
			zi := zWord[i]
			absZi := zi
			if absZi < 0 {
				absZi = -absZi
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

		if err != nil {
			return err
		}

		slices.SortFunc(wordMismatches, func(a, b wordMismatch) int {
			return cmp.Compare(a.absZi, b.absZi)
		})

		updateK := min(max(len(zWord)/d.GroupSize, 1), len(wordMismatches))
		for _, mismatch := range wordMismatches[:updateK] {
			tBit := mismatch.tBit
			col := mismatch.col
			deltaRow := delta[col*d.W.Cols() : (col+1)*d.W.Cols()]

			err = x.ScanRowsWord([]int{tCtx.Row}, func(xCtx bitsx.MatrixWordContext) error {
				xWord, err := x.Word(xCtx.WordIndex)
				if err != nil {
					return err
				}
				deltaWord := deltaRow[xCtx.ColStart:xCtx.ColEnd]
				for b := range deltaWord {
					xBit := (xWord >> uint(b)) & 1
					deltaWord[b] += int16(1 - 2*int(xBit^tBit))
				}
				return nil
			})

			if err != nil {
				return err
			}
		}
		return nil
	})
}

func (d *Dense) inputTarget(t *bitsx.Matrix) (*bitsx.Matrix, error) {
	uT, err := d.WT.Dot(t)
	if err != nil {
		return nil, err
	}
	tCols := t.Cols()

	xTarget, err := bitsx.NewZerosMatrix(t.Rows(), d.W.Cols())
	if err != nil {
		return nil, err
	}

	err = xTarget.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		var word uint64
		err := ctx.ScanBits(func(i, col, colT int) error {
			// 一致数 u が過半数かを見る。u >= tCols/2 と書くと、tCols が奇数のとき切り捨てで境界の符号が反転する
			if 2*uT[colT] >= tCols {
				word |= (1 << uint(i))
			}
			return nil
		})
		if err != nil {
			return err
		}
		return xTarget.SetWord(ctx.WordIndex, word)
	})

	if err != nil {
		return nil, err
	}
	return xTarget, nil
}

func (d *Dense) Predict(x *bitsx.Matrix) (*bitsx.Matrix, error) {
	z, err := d.preActivation(x)
	if err != nil {
		return nil, err
	}
	return bitsx.NewSignMatrix(x.Rows(), d.W.Rows(), z)
}

func (d *Dense) NewZerosDeltas() Deltas {
	n := d.W.Rows() * d.W.Cols()
	return Deltas{make(Delta, n)}
}

func (d *Dense) OutputShape(xRows, xCols int) (int, int, error) {
	if xCols != d.W.Cols() {
		return 0, 0, fmt.Errorf("入力の列数が不一致: xCols = %d, W.Cols = %d", xCols, d.W.Cols())
	}
	return xRows, d.W.Rows(), nil
}

func (d *Dense) Update(deltas Deltas, lrHalfPow int, rng *rand.Rand) error {
	if len(deltas) != 1 {
		return fmt.Errorf("deltasの数が不正: len(deltas) = %d: Dense層は1つのDeltaを持つべき", len(deltas))
	}

	delta := deltas[0]
	err := d.W.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		hWord := d.H[ctx.GlobalStart:ctx.GlobalEnd]
		deltaWord := delta[ctx.GlobalStart:ctx.GlobalEnd]

		// 要素ごとに乱数を引くより、1ワード分の更新対象をマスクでまとめて決める方が速いため。
		mask, err := bitsx.RandHalfPow[uint64](lrHalfPow, rng)
		if err != nil {
			return err
		}
		if ctx.IsTail {
			mask &= d.W.TailMask()
		}

		var flips uint64
		for mask != 0 {
			i := bits.TrailingZeros64(mask)
			mask &= mask - 1

			old := hWord[i]
			// オーバーフロー対策に一旦intにする
			newVal := int(old) + int(deltaWord[i])
			clipped := int8(max(math.MinInt8, min(newVal, math.MaxInt8)))
			hWord[i] = clipped

			oldIsNonNegative := old >= 0
			newIsNonNegative := clipped >= 0
			if oldIsNonNegative != newIsNonNegative {
				flips |= (1 << uint64(i))
				err := d.WT.Toggle(ctx.ColStart+i, ctx.Row)
				if err != nil {
					return err
				}
			}
		}

		old, err := d.W.Word(ctx.WordIndex)
		if err != nil {
			return err
		}
		return d.W.SetWord(ctx.WordIndex, old^flips)
	})
	return err
}

type Sequence []Layer

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

func (s Sequence) Update(seqDelta SeqDelta, lrHalfPow int, rngs []*rand.Rand) error {
	if len(s) != len(seqDelta) {
		return fmt.Errorf("層とSeqDeltaの数が不一致: len(Sequence) = %d, len(SeqDelta) = %d", len(s), len(seqDelta))
	}

	p := min(len(rngs), len(s))
	err := parallel.For(len(s), p, func(workerID, idx int) error {
		layer := s[idx]
		layerDelta := seqDelta[idx]
		rng := rngs[workerID]
		return layer.Update(layerDelta, lrHalfPow, rng)
	})
	return err
}

// 途中の層でエラーになったときに一部の層だけ変わるのを防ぐため、全層の値を計算してから代入する。
func (s Sequence) SetNoiseScale(num, denom int) error {
	maxAbsNoises := make([]int, len(s))
	for i, layer := range s {
		d, ok := layer.(*Dense)
		if !ok {
			continue
		}
		maxAbsNoise, err := maxAbsNoiseForScale(d.W.Cols(), num, denom)
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
		maxAbsNoises[i] = maxAbsNoise
	}
	for i, layer := range s {
		if d, ok := layer.(*Dense); ok {
			d.MaxAbsNoise = maxAbsNoises[i]
		}
	}
	return nil
}

func (s Sequence) SetGroupSize(groupSize int) error {
	if groupSize < 1 {
		return fmt.Errorf("GroupSizeが不正: GroupSize = %d: 1以上であるべき", groupSize)
	}
	for _, layer := range s {
		if d, ok := layer.(*Dense); ok {
			d.GroupSize = groupSize
		}
	}
	return nil
}

func init() {
	gob.Register(&Dense{})
}
