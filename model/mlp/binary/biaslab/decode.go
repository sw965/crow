package main

import (
	"math"
	"slices"

	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/parallel"
)

// 復号方式の比較。いずれも学習済みモデルの出力(全ラベルとのロジット)から値を作る。
//
// ロジットは「一致ビット数」なので、一致率 = ロジット / 総ビット数。
// どの方式も既に計算済みのロジットを使うだけで、追加の推論コストは無い。

// decodeTie は現行の PredictValue と同じ同点平均。
// 最大ロジットのラベルだけを使い、残りを捨てる。
func decodeTie(logits []int, values []float64) float64 {
	maxLogit := slices.Max(logits)
	var sum float64
	var count int
	for i, l := range logits {
		if l == maxLogit {
			sum += values[i]
			count++
		}
	}
	return sum / float64(count)
}

// decodeSoftmax は温度つき softmax 重みによる加重平均(../PROTOTYPES.md B-10b)。
func decodeSoftmax(logits []int, values []float64, temperature float64) float64 {
	maxLogit := slices.Max(logits)
	var num, den float64
	for i, l := range logits {
		w := math.Exp(float64(l-maxLogit) / temperature)
		num += w * values[i]
		den += w
	}
	return num / den
}

// decodeMatchRate は、一致率をそのまま重みにして合計1へ正規化する。
//
//	w_i = logit_i / Σ logit_j
//	答え = Σ w_i * value_i
//
// exp も学習パラメータも使わない、最も単純な加重平均。
func decodeMatchRate(logits []int, values []float64) float64 {
	var num, den float64
	for i, l := range logits {
		w := float64(l)
		num += w * values[i]
		den += w
	}
	return num / den
}

// decodeShiftedPow は、最小ロジットを引いてから p 乗した値を重みにする。
//
//	w_i = (logit_i - min_j logit_j)^p
//
// p=1 なら「最下位ラベルの重みを0にしてから正規化」。
// 一致率は 0.5 付近に密集するため、生の一致率をそのまま重みにすると
// 重みがほぼ均等になってしまう。最小値を引くと、その密集した台座が外れて
// ラベル間の差だけが残る。p を上げるとさらに鋭くなる。
func decodeShiftedPow(logits []int, values []float64, p int) float64 {
	minLogit := slices.Min(logits)
	var num, den float64
	for i, l := range logits {
		d := float64(l - minLogit)
		w := d
		for range p - 1 {
			w *= d
		}
		num += w * values[i]
		den += w
	}
	if den == 0 {
		// 全ラベルが同じロジット(情報が無い)。中央値相当を返す
		return values[len(values)/2]
	}
	return num / den
}

// decodeSpec は比較する復号方式ひとつ分。
// tune が非nilなら、候補パラメータを検証集合で選ぶ。
type decodeSpec struct {
	name string
	// fn は候補パラメータ(tune が nil なら 0)を受け取って復号する
	fn   func(logits []int, values []float64, param float64) float64
	tune []float64
}

func decodeSpecs() []decodeSpec {
	return []decodeSpec{
		{
			name: "同点平均(現行)",
			fn:   func(l []int, v []float64, _ float64) float64 { return decodeTie(l, v) },
		},
		{
			name: "提案: 一致率をそのまま正規化",
			fn:   func(l []int, v []float64, _ float64) float64 { return decodeMatchRate(l, v) },
		},
		{
			name: "提案+: 最小を引いて正規化 (p=1)",
			fn:   func(l []int, v []float64, _ float64) float64 { return decodeShiftedPow(l, v, 1) },
		},
		{
			name: "提案+: 最小を引いて2乗 (p=2)",
			fn:   func(l []int, v []float64, _ float64) float64 { return decodeShiftedPow(l, v, 2) },
		},
		{
			name: "提案+: 最小を引いてp乗 (pを検証で選ぶ)",
			fn:   func(l []int, v []float64, p float64) float64 { return decodeShiftedPow(l, v, int(p)) },
			tune: []float64{1, 2, 3, 4, 6, 8, 12, 16, 24, 32},
		},
		{
			name: "加重平均 softmax (Tを検証で選ぶ)",
			fn:   func(l []int, v []float64, t float64) float64 { return decodeSoftmax(l, v, t) },
			tune: []float64{5, 10, 20, 40, 60, 80, 120, 160, 240, 320, 480},
		},
	}
}

// maeWith は、与えた復号方式とパラメータでのMAEを返す。
func maeWith(spec decodeSpec, param float64, logits [][]int, levels []int, values []float64) float64 {
	var total float64
	for i, lg := range logits {
		total += math.Abs(spec.fn(lg, values, param) - values[levels[i]])
	}
	return total / float64(len(logits))
}

// evaluate は、検証集合でパラメータを選び、そのパラメータでのテストMAEを返す。
func (spec decodeSpec) evaluate(valLogits [][]int, valLevels []int,
	testLogits [][]int, testLevels []int, values []float64) (valMAE, testMAE, param float64) {
	if spec.tune == nil {
		v := maeWith(spec, 0, valLogits, valLevels, values)
		return v, maeWith(spec, 0, testLogits, testLevels, values), 0
	}
	bestP, bestVal := spec.tune[0], math.Inf(1)
	for _, p := range spec.tune {
		if v := maeWith(spec, p, valLogits, valLevels, values); v < bestVal {
			bestVal, bestP = v, p
		}
	}
	return bestVal, maeWith(spec, bestP, testLogits, testLevels, values), bestP
}

// logitSpread は、ロジットの散らばりを一致率の単位で返す(診断用)。
// 一致率が 0.5 付近に密集していると、生の一致率を重みにしても
// ほぼ均等な重みになってしまう。
func logitSpread(logits [][]int, totalBits int) (meanMax, meanMin, meanRange float64) {
	for _, lg := range logits {
		mx, mn := slices.Max(lg), slices.Min(lg)
		meanMax += float64(mx) / float64(totalBits)
		meanMin += float64(mn) / float64(totalBits)
		meanRange += float64(mx-mn) / float64(totalBits)
	}
	n := float64(len(logits))
	return meanMax / n, meanMin / n, meanRange / n
}

// ---------------------------------------------------------------------------
// 点灯数による復号(温度計符号の構造を直接使う)
// ---------------------------------------------------------------------------

// countDecode は、モデル出力からセルマスクを外して点灯数を数える。
//
// crow の温度計符号はレベル i で m_i = i*L/(n-1) ビットを立てるので、
// レベル0の符号は全ゼロである。したがってセルマスク M を掛けた後の
// prototypes[0] は M そのものになり、y XOR prototypes[0] でマスクが外れる。
// セルマスクを使っていない場合も prototypes[0] は全ゼロなので、同じ式が使える。
//
// プロトタイプとの距離を n 回計算する代わりに、XOR と popcount が1回で済む。
func countDecode(y *bitsx.Matrix, protoZero *bitsx.Matrix) (int, error) {
	unmasked, err := y.Xor(protoZero)
	if err != nil {
		return 0, err
	}
	return unmasked.OnesCount(), nil
}

// allCounts は全サンプルの点灯数を求める。
func (m *model) allCounts(xs bitsx.Matrices, p int) ([]int, error) {
	out := make([]int, len(xs))
	err := parallel.For(len(xs), p, func(workerID, i int) error {
		y, err := m.predict(xs[i])
		if err != nil {
			return err
		}
		c, err := countDecode(y, m.prototypes[0])
		if err != nil {
			return err
		}
		out[i] = c
		return nil
	})
	return out, err
}

// countLinearFit は、検証集合で 値 = a*点灯数 + b の最小二乗解を求める。
// 点灯数と値の関係が理論上は線形なので、系統的なずれ(オフセットや傾きの狂い)を
// これで測れる。校正パラメータは2個で、検証集合だけから決める。
func countLinearFit(counts []int, levels []int, values []float64) (a, b float64) {
	n := float64(len(counts))
	var sx, sy, sxx, sxy float64
	for i, c := range counts {
		x, y := float64(c), values[levels[i]]
		sx += x
		sy += y
		sxx += x * x
		sxy += x * y
	}
	den := n*sxx - sx*sx
	if den == 0 {
		return 0, sy / n
	}
	a = (n*sxy - sx*sy) / den
	b = (sy - a*sx) / n
	return a, b
}

// countMAE は、点灯数から値を作ったときのMAEを返す。
func countMAE(counts []int, levels []int, values []float64, a, b float64) float64 {
	lo, hi := values[0], values[len(values)-1]
	var total float64
	for i, c := range counts {
		v := a*float64(c) + b
		v = max(lo, min(v, hi)) // 値域でクリップ
		total += math.Abs(v - values[levels[i]])
	}
	return total / float64(len(counts))
}

// countStats は、点灯数から素直に作った予測値の平均と、真値の平均を返す。
// 系統的なずれ(ビット誤りの非対称性による点灯数の偏り)を見るための診断。
func countStats(counts []int, levels []int, values []float64, totalBits int) (meanPred, meanTrue float64) {
	lo, hi := values[0], values[len(values)-1]
	scale := (hi - lo) / float64(totalBits)
	for i, c := range counts {
		meanPred += lo + scale*float64(c)
		meanTrue += values[levels[i]]
	}
	n := float64(len(counts))
	return meanPred / n, meanTrue / n
}
