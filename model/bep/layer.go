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
	OutputShape(xRows, xCols int) (int, int, error)
	NewBatchRecord(batchSize, xRows int) (BatchRecord, error)
	BatchDeltas(rec BatchRecord) (Deltas, error)
	Update(deltas Deltas, lrHalfPow int, rng *rand.Rand) error
}

// サンプルごとにデルタを足し込む代わりに記録だけしておき、バッチの最後に BatchDeltas でまとめて計算するため。
type BatchRecord interface {
	Clear() error
}

type Backward func(t *bitsx.Matrix, rec BatchRecord, sampleIdx int) (*bitsx.Matrix, error)
type Backwards []Backward

func (bs Backwards) Propagate(t *bitsx.Matrix, recs []BatchRecord, sampleIdx int) (*bitsx.Matrix, error) {
	var err error
	for layerIdx := range slices.Backward(bs) {
		t, err = bs[layerIdx](t, recs[layerIdx], sampleIdx)
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

	Bias        []int32
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
		Bias:        make([]int32, wRows),
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
	// Bias が nil なのは、バイアス導入前に gob で保存したモデルを読み込んだ場合で、0 と同じに扱う
	if d.Bias != nil {
		yCols := d.W.Rows()
		for i := range z {
			z[i] += int(d.Bias[i%yCols])
		}
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

	backward := func(t *bitsx.Matrix, rec BatchRecord, sampleIdx int) (*bitsx.Matrix, error) {
		if err := t.ValidateSameShape(y); err != nil {
			return nil, err
		}
		r, ok := rec.(*denseBatchRecord)
		if !ok {
			return nil, fmt.Errorf("BatchRecordの型が不正: %T: Dense.NewBatchRecordで作成するべき", rec)
		}
		if err := d.recordSelection(x, z, t, r, sampleIdx); err != nil {
			return nil, err
		}
		return d.inputTarget(t)
	}
	return y, backward, nil
}

func (d *Dense) recordSelection(x *bitsx.Matrix, z []int, t *bitsx.Matrix, rec *denseBatchRecord, sampleIdx int) error {
	rowOffset := sampleIdx * x.Rows()
	if sampleIdx < 0 || rowOffset+x.Rows() > rec.x.Rows() {
		return fmt.Errorf("sampleIdxが範囲外: sampleIdx = %d: 記録できるのは %d 行まで", sampleIdx, rec.x.Rows())
	}
	if err := copyRows(rec.x, rowOffset, x); err != nil {
		return err
	}

	tStride := t.Stride()
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
		var nonZeroWord, signWord uint64
		for _, mismatch := range wordMismatches[:updateK] {
			i := uint(mismatch.col - tCtx.ColStart)
			nonZeroWord |= 1 << i
			signWord |= mismatch.tBit << i
		}
		idx := (rowOffset+tCtx.Row)*tStride + tCtx.ColStart/64
		if err := rec.nonZero.SetWord(idx, nonZeroWord); err != nil {
			return err
		}
		return rec.sign.SetWord(idx, signWord)
	})
}

// dst と src の列数(= ストライド)が同じである事を前提とする。
func copyRows(dst *bitsx.Matrix, dstRow int, src *bitsx.Matrix) error {
	stride := src.Stride()
	if dst.Stride() != stride {
		return fmt.Errorf("ストライドが不一致: dst = %d, src = %d", dst.Stride(), stride)
	}
	for i := range src.Rows() * stride {
		word, err := src.Word(i)
		if err != nil {
			return err
		}
		if err := dst.SetWord(dstRow*stride+i, word); err != nil {
			return err
		}
	}
	return nil
}

func clearMatrix(m *bitsx.Matrix) error {
	for i := range m.Rows() * m.Stride() {
		if err := m.SetWord(i, 0); err != nil {
			return err
		}
	}
	return nil
}

// 行は「サンプル番号 × 入力の行数 + 行」。
type denseBatchRecord struct {
	x       *bitsx.Matrix
	sign    *bitsx.Matrix
	nonZero *bitsx.Matrix
	ones    *bitsx.Matrix
	deltas  Deltas
}

func (r *denseBatchRecord) Clear() error {
	if err := clearMatrix(r.sign); err != nil {
		return err
	}
	return clearMatrix(r.nonZero)
}

func (d *Dense) NewBatchRecord(batchSize, xRows int) (BatchRecord, error) {
	rows := batchSize * xRows
	x, err := bitsx.NewZerosMatrix(rows, d.W.Cols())
	if err != nil {
		return nil, err
	}
	sign, err := bitsx.NewZerosMatrix(rows, d.W.Rows())
	if err != nil {
		return nil, err
	}
	nonZero, err := bitsx.NewZerosMatrix(rows, d.W.Rows())
	if err != nil {
		return nil, err
	}
	ones, err := bitsx.NewOnesMatrix(1, rows)
	if err != nil {
		return nil, err
	}
	return &denseBatchRecord{
		x:       x,
		sign:    sign,
		nonZero: nonZero,
		ones:    ones,
		deltas:  Deltas{make(Delta, d.W.Rows()*d.W.Cols()), make(Delta, d.W.Rows())},
	}, nil
}

// デルタ[j][b] = Σ_選択 (x_b と希望出力 t_j が一致なら +1、不一致なら −1) は、バッチ方向を内積の軸にした
// 三値(選択と向き)×二値(入力)の行列積なので、サンプルごとに int16 へ足し込む代わりに DotTernary 1回で求まる。
func (d *Dense) BatchDeltas(rec BatchRecord) (Deltas, error) {
	r, ok := rec.(*denseBatchRecord)
	if !ok {
		return nil, fmt.Errorf("BatchRecordの型が不正: %T: Dense.NewBatchRecordで作成するべき", rec)
	}
	xT, err := r.x.Transpose()
	if err != nil {
		return nil, err
	}
	signT, err := r.sign.Transpose()
	if err != nil {
		return nil, err
	}
	nonZeroT, err := r.nonZero.Transpose()
	if err != nil {
		return nil, err
	}

	sums, err := xT.DotTernary(signT, nonZeroT)
	if err != nil {
		return nil, err
	}
	fanIn := d.W.Cols()
	yCols := d.W.Rows()
	weightDelta := r.deltas[0]
	for b := range fanIn {
		for j := range yCols {
			weightDelta[j*fanIn+b] = clampInt16(sums[b*yCols+j])
		}
	}

	// 全ビット1の値と比べると、選んだ位置の (希望出力が1なら+1、0なら−1) の合計になる
	biasSums, err := r.ones.DotTernary(signT, nonZeroT)
	if err != nil {
		return nil, err
	}
	biasDelta := r.deltas[1]
	for j := range yCols {
		biasDelta[j] = clampInt16(biasSums[j])
	}
	return r.deltas, nil
}

// デルタは最終的に符号しか使わないため、int16 に収まらない大きさは符号を保ったまま切り詰める
func clampInt16(v int) int16 {
	return int16(max(math.MinInt16, min(v, math.MaxInt16)))
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

func (d *Dense) OutputShape(xRows, xCols int) (int, int, error) {
	if xCols != d.W.Cols() {
		return 0, 0, fmt.Errorf("入力の列数が不一致: xCols = %d, W.Cols = %d", xCols, d.W.Cols())
	}
	return xRows, d.W.Rows(), nil
}

func (d *Dense) Update(deltas Deltas, lrHalfPow int, rng *rand.Rand) error {
	if len(deltas) != 2 {
		return fmt.Errorf("deltasの数が不正: len(deltas) = %d: Dense層は重みとバイアスの2つのDeltaを持つべき", len(deltas))
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
	if err != nil {
		return err
	}
	// 重みの後に更新するのは、Bias が 0 のままなら重みの更新が乱数の消費まで導入前と一致するようにするため
	return d.updateBias(deltas[1], lrHalfPow, rng)
}

func (d *Dense) updateBias(biasDelta Delta, lrHalfPow int, rng *rand.Rand) error {
	if d.Bias == nil {
		d.Bias = make([]int32, d.W.Rows())
	}
	// |Bias| が fanIn を超えると、入力によらず出力が一定になり、それ以上動かしても意味がないため
	maxAbsBias := int32(d.W.Cols())
	for start := 0; start < len(d.Bias); start += 64 {
		mask, err := bitsx.RandHalfPow[uint64](lrHalfPow, rng)
		if err != nil {
			return err
		}
		if n := len(d.Bias) - start; n < 64 {
			mask &= (uint64(1) << uint(n)) - 1
		}
		for mask != 0 {
			j := start + bits.TrailingZeros64(mask)
			mask &= mask - 1
			d.Bias[j] = max(-maxAbsBias, min(d.Bias[j]+int32(biasDelta[j]), maxAbsBias))
		}
	}
	return nil
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
