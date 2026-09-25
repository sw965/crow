package bep

import (
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

type Layer interface {
	Forward(x *bitsx.Matrix) (*bitsx.Matrix, Backward, error)
	Predict(x *bitsx.Matrix) (*bitsx.Matrix, error)
	OutputShape(xRows, xCols int) (int, int, error)
	NewBatchRecord(batchSize, xRows int) (BatchRecord, error)
	BatchDeltas(rec BatchRecord) (Deltas, error)
	Update(deltas Deltas, lrHalfPow int, rng *rand.Rand) error
	Validate() error
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

	Bias          []int32
	MaxUpdateAbsZ int
	MarginAbsZ    int
}

func isqrt(n int) int {
	r := 0
	for (r+1)*(r+1) <= n {
		r++
	}
	return r
}

// 倍率 num/denom は、z の自然なばらつき √fanIn の何倍にするかを表す。
// 上限は |z| の最大値 2 × fanIn(Bias の上限 fanIn を含む)。これを超えても最大値と変わらない。
// 極端な num で掛け算が桁あふれしないよう big.Int で計算する。
func absZForScale(fanIn, num, denom int) (int, error) {
	if num < 0 || denom <= 0 {
		return 0, fmt.Errorf("倍率が不正: num = %d, denom = %d: num >= 0 かつ denom > 0 であるべき", num, denom)
	}
	v := new(big.Int).Mul(big.NewInt(int64(isqrt(fanIn))), big.NewInt(int64(num)))
	v.Quo(v, big.NewInt(int64(denom)))
	// 入力数が小さいと切り捨てで 0 になり、MaxUpdateAbsZ は Validate を通らず、MarginAbsZ はマージンが効かなくなる
	// (入力数 49 の畳み込みで、マージンが 0 になり学習できなかった。CNN.md §2)。正の倍率なら最低 1 にする
	if num > 0 && v.Sign() == 0 {
		v.SetInt64(1)
	}
	if !v.IsInt64() || v.Int64() > int64(2*fanIn) {
		return 0, fmt.Errorf("倍率が大きすぎる: num/denom = %d/%d: |z| の最大値 %d を超える", num, denom, 2*fanIn)
	}
	return int(v.Int64()), nil
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

	// 倍率は MaxUpdateAbsZ・MarginAbsZ とも 1/4。旧方式(64 個ごとに |z| で並べ替えて上位 16 個を直し、z にノイズを加える)より
	// MNIST で +2.8pt(90.8 → 93.6%)、Fashion-MNIST で +1.8pt(83.1 → 84.8%)良かった(30エポック・3シード)。
	// MarginAbsZ は 3/4 以上にすると正解側を直しすぎて崩れた(MNIST で 89.5%)。
	maxUpdateAbsZ, err := absZForScale(w.Cols(), 1, 4)
	if err != nil {
		return nil, err
	}
	marginAbsZ, err := absZForScale(w.Cols(), 1, 4)
	if err != nil {
		return nil, err
	}

	return &Dense{
		W:             w,
		WT:            wt,
		H:             h,
		Bias:          make([]int32, wRows),
		MaxUpdateAbsZ: maxUpdateAbsZ,
		MarginAbsZ:    marginAbsZ,
	}, nil
}

func setAbsZScale(dst *int, name string, fanIn, num, denom int) error {
	absZ, err := absZForScale(fanIn, num, denom)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	*dst = absZ
	return nil
}

func (d *Dense) SetMaxUpdateAbsZScale(num, denom int) error {
	return setAbsZScale(&d.MaxUpdateAbsZ, "MaxUpdateAbsZ", d.W.Cols(), num, denom)
}

func (d *Dense) SetMarginAbsZScale(num, denom int) error {
	return setAbsZScale(&d.MarginAbsZ, "MarginAbsZ", d.W.Cols(), num, denom)
}

func (d *Dense) preActivation(x *bitsx.Matrix) ([]int, error) {
	u, err := x.Dot(d.W)
	if err != nil {
		return nil, err
	}

	maxZi := d.W.Cols()
	// u はここでしか使わないので、新しく確保せずそのまま z として書き換える(学習中はサンプルごとに呼ばれるため)
	z := u
	for i, count := range u {
		z[i] = 2*count - maxZi
	}
	yCols := d.W.Rows()
	for rowStart := 0; rowStart < len(z); rowStart += yCols {
		zRow := z[rowStart : rowStart+yCols]
		for j := range zRow {
			zRow[j] += int(d.Bias[j])
		}
	}
	return z, nil
}

func (d *Dense) Forward(x *bitsx.Matrix) (*bitsx.Matrix, Backward, error) {
	z, err := d.preActivation(x)
	if err != nil {
		return nil, nil, err
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
		rowOffset, err := r.recordInput(x, sampleIdx)
		if err != nil {
			return nil, err
		}
		if err := d.recordSelection(z, y, t, r, rowOffset); err != nil {
			return nil, err
		}
		votes, err := d.inputVotes(t)
		if err != nil {
			return nil, err
		}
		return targetFromVotes(t.Rows(), d.W.Cols(), votes)
	}
	return y, backward, nil
}

func (d *Dense) recordSelection(z []int, y, t *bitsx.Matrix, rec *denseBatchRecord, rowOffset int) error {
	return t.ScanRowsWord(nil, func(tCtx bitsx.MatrixWordContext) error {
		// 64ビット毎に操作するための宣言
		zWord := z[tCtx.GlobalStart:tCtx.GlobalEnd]

		tWord, err := t.Word(tCtx.WordIndex)
		if err != nil {
			return err
		}
		yWord, err := y.Word(tCtx.WordIndex)
		if err != nil {
			return err
		}

		mismatchMask := yWord ^ tWord
		var nonZeroWord uint64
		for i, zi := range zWord {
			bit := uint64(1) << uint(i)
			absZ := max(zi, -zi)
			// 食い違っていても |z| が大きいニューロンは、反転に大きな修正が要り、他の入力への影響も大きいので直さない。
			// 正解していても |z| が小さいニューロンは、入力の少しの違いや他の更新ですぐ反転するので、目標の向きへ押して余裕を持たせる
			// (マージン。旧実装で z に加えていたノイズは、これを確率的に行っていた)。
			isMismatch := mismatchMask&bit != 0
			if (isMismatch && absZ <= d.MaxUpdateAbsZ) || (!isMismatch && absZ < d.MarginAbsZ) {
				nonZeroWord |= bit
			}
		}
		// 食い違いでもマージンでも、押す向きは目標 t
		return rec.setSelection(rowOffset+tCtx.Row, tCtx.ColStart, nonZeroWord, tWord&nonZeroWord)
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

func (r *denseBatchRecord) recordInput(x *bitsx.Matrix, sampleIdx int) (int, error) {
	rowOffset := sampleIdx * x.Rows()
	if sampleIdx < 0 || rowOffset+x.Rows() > r.x.Rows() {
		return 0, fmt.Errorf("sampleIdxが範囲外: sampleIdx = %d: 記録できるのは %d 行まで", sampleIdx, r.x.Rows())
	}
	if err := copyRows(r.x, rowOffset, x); err != nil {
		return 0, err
	}
	return rowOffset, nil
}

func (r *denseBatchRecord) setSelection(row, colStart int, nonZeroWord, signWord uint64) error {
	idx := row*r.sign.Stride() + colStart/64
	if err := r.nonZero.SetWord(idx, nonZeroWord); err != nil {
		return err
	}
	return r.sign.SetWord(idx, signWord)
}

func (d *Dense) NewBatchRecord(batchSize, xRows int) (BatchRecord, error) {
	x, err := bitsx.NewZerosMatrix(batchSize*xRows, d.W.Cols())
	if err != nil {
		return nil, err
	}
	rec, err := d.newBatchRecord(x)
	if err != nil {
		return nil, err
	}
	return rec, nil
}

func (d *Dense) newBatchRecord(x *bitsx.Matrix) (*denseBatchRecord, error) {
	rows := x.Rows()
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

// 票は「一致数 u − 不一致数」= 2u − tCols で表す。u >= tCols/2 と比べると、tCols が奇数のとき
// 切り捨てで境界の符号が反転するため。また、差の形なら ProductDense で枝ごとの票をそのまま足せる。
// 並びは WT.Dot の結果のまま(転置の添字 colT)。
func (d *Dense) inputVotes(t *bitsx.Matrix) ([]int, error) {
	votes, err := d.WT.Dot(t)
	if err != nil {
		return nil, err
	}
	tCols := t.Cols()
	for i, u := range votes {
		votes[i] = 2*u - tCols
	}
	return votes, nil
}

func targetFromVotes(rows, cols int, votes []int) (*bitsx.Matrix, error) {
	xTarget, err := bitsx.NewZerosMatrix(rows, cols)
	if err != nil {
		return nil, err
	}

	err = xTarget.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		var word uint64
		err := ctx.ScanBits(func(i, col, colT int) error {
			if votes[colT] >= 0 {
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

func (d *Dense) Validate() error {
	if len(d.Bias) != d.W.Rows() {
		return fmt.Errorf("biasの長さが不正: len(Bias) = %d: 出力数 %d であるべき", len(d.Bias), d.W.Rows())
	}
	for j, b := range d.Bias {
		if b < -int32(d.W.Cols()) || b > int32(d.W.Cols()) {
			return fmt.Errorf("bias[%d]が不正: Bias = %d: |Bias| <= %d (入力数) であるべき", j, b, d.W.Cols())
		}
	}
	return validateAbsZ(d.MaxUpdateAbsZ, d.MarginAbsZ, d.W.Cols())
}

func validateAbsZ(maxUpdateAbsZ, marginAbsZ, fanIn int) error {
	// 0 だと |z| = 0 のニューロンしか直さず、学習がほぼ止まるため
	if maxUpdateAbsZ < 1 || maxUpdateAbsZ > 2*fanIn {
		return fmt.Errorf("MaxUpdateAbsZが不正: MaxUpdateAbsZ = %d: 1 <= MaxUpdateAbsZ <= %d (|z| の最大値) であるべき", maxUpdateAbsZ, 2*fanIn)
	}
	if marginAbsZ < 0 || marginAbsZ > 2*fanIn {
		return fmt.Errorf("MarginAbsZが不正: MarginAbsZ = %d: 0 <= MarginAbsZ <= %d (|z| の最大値) であるべき", marginAbsZ, 2*fanIn)
	}
	return nil
}

// 枝の符号の積にするのは、1ユニットで超平面1枚では分けられない関数も表せ、層を重ねずに表現力を稼げるため。
// 通常の層を深くしても精度が上がらない BEP で、積の層は MNIST の精度を大きく上げた(MULTIPLICATIVE_UNIT.md §5-5)。
type ProductDense struct {
	Branches []*Dense
	// 枝の MaxUpdateAbsZ・MarginAbsZ は使わない。直すかどうかは、どの枝を直すかを選ぶ前に、枝の |z| の最小値で層全体として決まるため。
	MaxUpdateAbsZ int
	MarginAbsZ    int
}

func NewProductDense(wRows, wCols, numBranches int, rng *rand.Rand) (*ProductDense, error) {
	// 1本では Dense と同じ層になり、3本以上は MNIST の実測で2本より精度が下がった(MULTIPLICATIVE_UNIT.md §5-5)
	if numBranches < 2 {
		return nil, fmt.Errorf("枝の数が不正: numBranches = %d: 2以上であるべき", numBranches)
	}
	branches := make([]*Dense, numBranches)
	for b := range branches {
		d, err := NewDense(wRows, wCols, rng)
		if err != nil {
			return nil, err
		}
		branches[b] = d
	}
	// 倍率 1/2 は、旧方式(GroupSize = 4)より MNIST で +0.4pt(94.8 → 95.3%)、Fashion-MNIST で
	// +0.5pt(84.2 → 84.7%)良かった(30エポック・3シード)。Dense と同じ 1/4 では MNIST の10エポックで 93.5% に下がった。
	// Dense より大きいのは、1つの食い違いを枝1本でしか直さず、1回の修正が弱いためと考えられる。
	maxUpdateAbsZ, err := absZForScale(wCols, 1, 2)
	if err != nil {
		return nil, err
	}
	// 倍率 1/8 は、マージン無しより MNIST で +0.1pt(95.2 → 95.3%)、Fashion-MNIST で +0.2pt(84.6 → 84.8%)、
	// 回帰の MAE も全タスクで小さかった(30エポック・3シード)。Dense と同じ 1/4 は MNIST で逆効果(94.9%)だった。
	marginAbsZ, err := absZForScale(wCols, 1, 8)
	if err != nil {
		return nil, err
	}
	return &ProductDense{Branches: branches, MaxUpdateAbsZ: maxUpdateAbsZ, MarginAbsZ: marginAbsZ}, nil
}

func (p *ProductDense) SetMaxUpdateAbsZScale(num, denom int) error {
	return setAbsZScale(&p.MaxUpdateAbsZ, "MaxUpdateAbsZ", p.Branches[0].W.Cols(), num, denom)
}

func (p *ProductDense) SetMarginAbsZScale(num, denom int) error {
	return setAbsZScale(&p.MarginAbsZ, "MarginAbsZ", p.Branches[0].W.Cols(), num, denom)
}

func (p *ProductDense) preActivations(x *bitsx.Matrix) ([][]int, error) {
	zs := make([][]int, len(p.Branches))
	for b, d := range p.Branches {
		z, err := d.preActivation(x)
		if err != nil {
			return nil, err
		}
		zs[b] = z
	}
	return zs, nil
}

func (p *ProductDense) branchSigns(xRows int, zs [][]int) ([]*bitsx.Matrix, error) {
	signs := make([]*bitsx.Matrix, len(zs))
	for b, z := range zs {
		sign, err := bitsx.NewSignMatrix(xRows, p.Branches[b].W.Rows(), z)
		if err != nil {
			return nil, err
		}
		signs[b] = sign
	}
	return signs, nil
}

// 符号ビット(1 が +1)の XNOR が ±1 の積に当たるので、ニューロンごとに掛けずに 64 個ずつワードでまとめて求める。
func output(signs []*bitsx.Matrix) (*bitsx.Matrix, error) {
	y := signs[0].Clone()
	for _, sign := range signs[1:] {
		for i := range y.Rows() * y.Stride() {
			yWord, err := y.Word(i)
			if err != nil {
				return nil, err
			}
			signWord, err := sign.Word(i)
			if err != nil {
				return nil, err
			}
			if err := y.SetWord(i, ^(yWord ^ signWord)); err != nil {
				return nil, err
			}
		}
	}
	return y, nil
}

func (p *ProductDense) Forward(x *bitsx.Matrix) (*bitsx.Matrix, Backward, error) {
	zs, err := p.preActivations(x)
	if err != nil {
		return nil, nil, err
	}

	signs, err := p.branchSigns(x.Rows(), zs)
	if err != nil {
		return nil, nil, err
	}
	y, err := output(signs)
	if err != nil {
		return nil, nil, err
	}

	backward := func(t *bitsx.Matrix, rec BatchRecord, sampleIdx int) (*bitsx.Matrix, error) {
		if err := t.ValidateSameShape(y); err != nil {
			return nil, err
		}
		r, ok := rec.(*productBatchRecord)
		if !ok {
			return nil, fmt.Errorf("BatchRecordの型が不正: %T: ProductDense.NewBatchRecordで作成するべき", rec)
		}
		// 入力は全枝で共有しているので、書き込みは1回でよい
		rowOffset, err := r.branches[0].recordInput(x, sampleIdx)
		if err != nil {
			return nil, err
		}
		branchTargets, err := p.recordSelection(zs, signs, y, t, r, rowOffset)
		if err != nil {
			return nil, err
		}

		fanIn := p.Branches[0].W.Cols()
		var votes []int
		for b, d := range p.Branches {
			branchVotes, err := d.inputVotes(branchTargets[b])
			if err != nil {
				return nil, err
			}
			if votes == nil {
				votes = branchVotes
				continue
			}
			for i, v := range branchVotes {
				votes[i] += v
			}
		}
		return targetFromVotes(t.Rows(), fanIn, votes)
	}
	return y, backward, nil
}

// 食い違ったニューロンで反転させる枝を1本に絞るのは、2本以上を反転させると積が元に戻るため(MULTIPLICATIVE_UNIT.md §2-1)。
// 枝はバッチ単位で交互に選ぶと更新が打ち消し合って学習しないため、ニューロンごとに |z| が最小の(最も直しやすい)枝を選ぶ(§2-2)。
// signs は Forward の出力と共有しているので、書き換えずに複製してから反転させる(backward は何度でも呼べるため)。
func (p *ProductDense) recordSelection(zs [][]int, signs []*bitsx.Matrix, y, t *bitsx.Matrix, rec *productBatchRecord, rowOffset int) ([]*bitsx.Matrix, error) {
	numBranches := len(p.Branches)
	branchTargets := make([]*bitsx.Matrix, numBranches)
	for b, sign := range signs {
		branchTargets[b] = sign.Clone()
	}

	flipWords := make([]uint64, numBranches)
	nonZeroWords := make([]uint64, numBranches)
	signWords := make([]uint64, numBranches)
	err := t.ScanRowsWord(nil, func(tCtx bitsx.MatrixWordContext) error {
		tWord, err := t.Word(tCtx.WordIndex)
		if err != nil {
			return err
		}
		yWord, err := y.Word(tCtx.WordIndex)
		if err != nil {
			return err
		}
		clear(flipWords)
		clear(nonZeroWords)
		clear(signWords)

		mismatchMask := yWord ^ tWord
		for i := range tCtx.ColEnd - tCtx.ColStart {
			idx := tCtx.GlobalStart + i
			cheapest := 0
			minAbsZ := math.MaxInt
			for b, z := range zs {
				if absZ := max(z[idx], -z[idx]); absZ < minAbsZ {
					cheapest, minAbsZ = b, absZ
				}
			}

			bit := uint64(1) << uint(i)
			isMismatch := mismatchMask&bit != 0
			if isMismatch {
				// 枝への希望出力は、直すかどうかに関わらず反転させる。前の層への票は、出力を t に合わせる向きで数えるため
				flipWords[cheapest] |= bit
				if minAbsZ > p.MaxUpdateAbsZ {
					continue
				}
			} else if minAbsZ >= p.MarginAbsZ {
				continue
			}
			nonZeroWords[cheapest] |= bit
			// 食い違いなら枝の今の符号の逆へ直し、マージンなら今の符号のまま境界から遠ざける
			if (zs[cheapest][idx] >= 0) != isMismatch {
				signWords[cheapest] |= bit
			}
		}

		for b, target := range branchTargets {
			old, err := target.Word(tCtx.WordIndex)
			if err != nil {
				return err
			}
			if err := target.SetWord(tCtx.WordIndex, old^flipWords[b]); err != nil {
				return err
			}
		}

		for b, r := range rec.branches {
			if err := r.setSelection(rowOffset+tCtx.Row, tCtx.ColStart, nonZeroWords[b], signWords[b]); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return branchTargets, nil
}

// 入力 x は全枝で同じなので、枝ごとの記録に複製せず1つの行列を共有する(書き込みとメモリを枝の数だけ増やさないため)。
type productBatchRecord struct {
	branches []*denseBatchRecord
}

func (r *productBatchRecord) Clear() error {
	for _, br := range r.branches {
		if err := br.Clear(); err != nil {
			return err
		}
	}
	return nil
}

func (p *ProductDense) NewBatchRecord(batchSize, xRows int) (BatchRecord, error) {
	x, err := bitsx.NewZerosMatrix(batchSize*xRows, p.Branches[0].W.Cols())
	if err != nil {
		return nil, err
	}
	branches := make([]*denseBatchRecord, len(p.Branches))
	for b, d := range p.Branches {
		br, err := d.newBatchRecord(x)
		if err != nil {
			return nil, err
		}
		branches[b] = br
	}
	return &productBatchRecord{branches: branches}, nil
}

func (p *ProductDense) BatchDeltas(rec BatchRecord) (Deltas, error) {
	r, ok := rec.(*productBatchRecord)
	if !ok {
		return nil, fmt.Errorf("BatchRecordの型が不正: %T: ProductDense.NewBatchRecordで作成するべき", rec)
	}
	deltas := make(Deltas, 0, 2*len(p.Branches))
	for b, d := range p.Branches {
		branchDeltas, err := d.BatchDeltas(r.branches[b])
		if err != nil {
			return nil, err
		}
		deltas = append(deltas, branchDeltas...)
	}
	return deltas, nil
}

func (p *ProductDense) Update(deltas Deltas, lrHalfPow int, rng *rand.Rand) error {
	if len(deltas) != 2*len(p.Branches) {
		return fmt.Errorf("deltasの数が不正: len(deltas) = %d: 枝ごとに重みとバイアスの2つ、計 %d 個であるべき", len(deltas), 2*len(p.Branches))
	}
	for b, d := range p.Branches {
		if err := d.Update(deltas[2*b:2*b+2], lrHalfPow, rng); err != nil {
			return err
		}
	}
	return nil
}

func (p *ProductDense) Predict(x *bitsx.Matrix) (*bitsx.Matrix, error) {
	zs, err := p.preActivations(x)
	if err != nil {
		return nil, err
	}
	signs, err := p.branchSigns(x.Rows(), zs)
	if err != nil {
		return nil, err
	}
	return output(signs)
}

func (p *ProductDense) OutputShape(xRows, xCols int) (int, int, error) {
	return p.Branches[0].OutputShape(xRows, xCols)
}

func (p *ProductDense) Validate() error {
	if len(p.Branches) < 2 {
		return fmt.Errorf("branchesの数が不正: len(Branches) = %d: 2以上であるべき", len(p.Branches))
	}
	first := p.Branches[0]
	for b, d := range p.Branches {
		if d == nil {
			return fmt.Errorf("branches[%d]がnilです", b)
		}
		if d.W.Rows() != first.W.Rows() || d.W.Cols() != first.W.Cols() {
			return fmt.Errorf("branches[%d]の形状が不一致: (%d, %d): Branches[0] と同じ (%d, %d) であるべき",
				b, d.W.Rows(), d.W.Cols(), first.W.Rows(), first.W.Cols())
		}
		if err := d.Validate(); err != nil {
			return fmt.Errorf("branches[%d]: %w", b, err)
		}
	}
	return validateAbsZ(p.MaxUpdateAbsZ, p.MarginAbsZ, first.W.Cols())
}

type Sequence []Layer

func (s Sequence) Forward(x *bitsx.Matrix) (*bitsx.Matrix, Backwards, error) {
	var backward Backward
	var err error
	backwards := make(Backwards, len(s))
	for i, layer := range s {
		x, backward, err = layer.Forward(x)
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

func (s Sequence) SetMaxUpdateAbsZScale(num, denom int) error {
	return s.setAbsZScale("MaxUpdateAbsZ", num, denom, func(maxUpdateAbsZ, _ *int) *int { return maxUpdateAbsZ })
}

func (s Sequence) SetMarginAbsZScale(num, denom int) error {
	return s.setAbsZScale("MarginAbsZ", num, denom, func(_, marginAbsZ *int) *int { return marginAbsZ })
}

// 途中の層でエラーになったときに一部の層だけ変わるのを防ぐため、全層の値を計算してから代入する。
func (s Sequence) setAbsZScale(name string, num, denom int, field func(maxUpdateAbsZ, marginAbsZ *int) *int) error {
	absZs := make([]int, len(s))
	for i, layer := range s {
		fanIn, _, _, ok := absZFields(layer)
		if !ok {
			continue
		}
		absZ, err := absZForScale(fanIn, num, denom)
		if err != nil {
			return fmt.Errorf("layer %d: %s: %w", i, name, err)
		}
		absZs[i] = absZ
	}
	for i, layer := range s {
		if _, maxUpdateAbsZ, marginAbsZ, ok := absZFields(layer); ok {
			*field(maxUpdateAbsZ, marginAbsZ) = absZs[i]
		}
	}
	return nil
}

func absZFields(layer Layer) (fanIn int, maxUpdateAbsZ, marginAbsZ *int, ok bool) {
	switch l := layer.(type) {
	case *Dense:
		return l.W.Cols(), &l.MaxUpdateAbsZ, &l.MarginAbsZ, true
	case *ProductDense:
		return l.Branches[0].W.Cols(), &l.MaxUpdateAbsZ, &l.MarginAbsZ, true
	}
	return 0, nil, nil, false
}

func init() {
	gob.Register(&Dense{})
	gob.Register(&ProductDense{})
}
