// swalab: BEP に SWA(重み平均)を適用する実験。
//
// 依頼された手順:
//
//  1. モデルAを学習する
//  2. モデルBを学習し、AとSWAして C とする
//  3. モデルDを学習し、CとSWAする … を繰り返す
//
// 二値ネットで「重みを平均する」には解釈が要る。可視重み W は ±1 なので直接は
// 平均できない。BEP は整数の隠れ重み H を持ち W = sign(H) なので、
// **H を平均して W = sign(平均H) を取り直す**のが素直な対応になる
// (QAT の潜在重みに対する SWA と同じ位置づけ)。本実験はこれを使う。
// 平均は整数の総和と整数除算だけで行い、浮動小数点は使わない。
//
// ただし独立に初期化したモデル同士の重み平均は、一般に**順列対称性**のせいで壊れる。
// モデルAのニューロン5とモデルBのニューロン5は無関係な特徴を計算しているためである。
// 通常の SWA が機能するのは、同一の学習軌跡上の点(=対応が取れている重み)を
// 平均するからである。
//
// そこで3つの解釈を比較する (-mode)。
//
//	independent : 毎回ゼロから初期化して学習し、平均する(依頼の手順どおり)
//	continued   : 現在の平均モデルから学習を再開し、その結果を平均に加える
//	trajectory  : 1本の学習を続けながら E エポックごとにスナップショットを平均する(本来のSWA)
//
// どのモードでもプロトタイプ P は全モデルで共有する(P が違うと目標符号が変わり、
// 重みを平均する意味が無くなるため)。
//
// BEP の順伝播・逆伝播・更新は ../layer.go ../train.go と同じ手順を bitsx の
// 公開APIで再実装したもので、../biaslab ../e2elab と同じ土台を使っている
// (crow 本体との1ステップ差分テストは ../biaslab/main_test.go にある)。
// crow / omw のライブラリコードは変更していない。
//
// 実行例:
//
//	go run . -mode independent -models 5 -epochs 20
//	go run . -mode trajectory  -models 5 -epochs 20
package main

import (
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/mathx/randx"
)

const (
	hInitAbs   = 4 // ../layer.go と同じ
	numClasses = 10
)

// ---------------------------------------------------------------------------
// デルタ
// ---------------------------------------------------------------------------

type delta struct {
	w []int16 // wRows * wCols
}

func newDelta(wRows, wCols int) *delta { return &delta{w: make([]int16, wRows*wCols)} }

func (d *delta) clear() { clear(d.w) }

func (d *delta) add(other *delta) {
	for i, v := range other.w {
		d.w[i] += v
	}
}

func (d *delta) sign() {
	for i, v := range d.w {
		d.w[i] = int16(cmp.Compare(v, 0))
	}
}

// ---------------------------------------------------------------------------
// Dense層(../layer.go の再実装)
// ---------------------------------------------------------------------------

type dense struct {
	w  *bitsx.Matrix
	wt *bitsx.Matrix
	h  []int8

	gateBase  int
	noiseStd  float32
	gateScale float32
	groupSize int
}

func newDense(wRows, wCols int, rng *rand.Rand) (*dense, error) {
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
		gateBase: int(noiseStdBase), noiseStd: noiseStdBase,
		gateScale: 1.0, groupSize: 4,
	}, nil
}

func (d *dense) outputShape(xRows, xCols int) (int, int, error) {
	if xCols != d.w.Cols() {
		return 0, 0, fmt.Errorf("入力の列数が不一致: xCols = %d, W.Cols = %d", xCols, d.w.Cols())
	}
	return xRows, d.w.Rows(), nil
}

func (d *dense) newDelta() *delta { return newDelta(d.w.Rows(), d.w.Cols()) }

func (d *dense) preActivation(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) ([]int, int, int, error) {
	u, err := x.Dot(d.w)
	if err != nil {
		return nil, 0, 0, err
	}
	fanIn := d.w.Cols()
	z := make([]int, len(u))
	std := noiseScale * d.noiseStd
	for i, count := range u {
		zi := 2*count - fanIn
		if std > 0 {
			noise, err := randx.IntNorm(-fanIn, fanIn, 0, std, rng)
			if err != nil {
				return nil, 0, 0, err
			}
			zi += noise
		}
		z[i] = zi
	}
	return z, x.Rows(), d.w.Rows(), nil
}

// backward は希望活性 t を受け取り、前段への希望活性を返す。
// care が非nilなら、そのビットが0の位置はゲートから外し、
// 後方射影 sign(W^T (g ⊙ t)) に寄与させない。
// 「その位置は指定なし」を、局所の更新対象からだけでなく
// 前段への信号からも取り除くために使う。
type backward func(t *bitsx.Matrix, care *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error)

func (d *dense) forward(x *bitsx.Matrix, noiseScale float32, rng *rand.Rand) (*bitsx.Matrix, backward, error) {
	z, yRows, yCols, err := d.preActivation(x, noiseScale, rng)
	if err != nil {
		return nil, nil, err
	}
	y, err := bitsx.NewSignMatrix(yRows, yCols, z)
	if err != nil {
		return nil, nil, err
	}

	bw := func(t *bitsx.Matrix, care *bitsx.Matrix, dl *delta) (*bitsx.Matrix, error) {
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
			if care != nil {
				careWord, err := care.Word(tCtx.WordIndex)
				if err != nil {
					return err
				}
				keepGateWord &= careWord
			}
			if err := keepGate.SetWord(tCtx.WordIndex, keepGateWord); err != nil {
				return err
			}
			slices.SortFunc(mismatches, func(a, b wordMismatch) int {
				return cmp.Compare(a.absZi, b.absZi)
			})
			updateK := min(max(len(zWord)/d.groupSize, 1), len(mismatches))

			for _, mm := range mismatches[:updateK] {
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

func (d *dense) update(dl *delta, lr float32, rng *rand.Rand) error {
	return d.w.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
		hWord := d.h[ctx.GlobalStart:ctx.GlobalEnd]
		dWord := dl.w[ctx.GlobalStart:ctx.GlobalEnd]
		var flips uint64
		if err := ctx.ScanBits(func(i, col, colT int) error {
			if rng.Float32() > lr {
				return nil
			}
			old := hWord[i]
			clipped := int8(max(math.MinInt8, min(int(old)+int(dWord[i]), math.MaxInt8)))
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
