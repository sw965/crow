package main

import (
	"errors"

	"github.com/sw965/omw/mathx/bitsx"
	"github.com/sw965/omw/parallel"
)

// ---------------------------------------------------------------------------
// アンサンブル
// ---------------------------------------------------------------------------

// ensemble は複数モデルのロジット(プロトタイプとの一致ビット数)を合計して予測する。
//
// SWA と違い**重みを混ぜない**ので、独立に初期化したモデルでも成立する
// (順列対称性の問題を受けない)。プロトタイプは全モデルで共有しているため、
// ロジットは同じ尺度の整数であり、そのまま足せる。
type ensemble struct {
	models []*model
}

func (e *ensemble) add(m *model) { e.models = append(e.models, m) }

// logits は各モデルのロジットの合計を返す。整数加算のみ。
func (e *ensemble) logits(x *bitsx.Matrix) ([]int, error) {
	var sum []int
	for _, m := range e.models {
		y, err := m.predict(x)
		if err != nil {
			return nil, err
		}
		lg, err := m.logits(y)
		if err != nil {
			return nil, err
		}
		if sum == nil {
			sum = make([]int, len(lg))
		}
		for i, v := range lg {
			sum[i] += v
		}
	}
	return sum, nil
}

func argmaxInt(v []int) int {
	best, bestVal := 0, v[0]
	for i, x := range v {
		if x > bestVal {
			bestVal, best = x, i
		}
	}
	return best
}

func (e *ensemble) accuracy(xs bitsx.Matrices, labels []int, p int) (float64, error) {
	counts := make([]int, p)
	err := parallel.For(len(xs), p, func(workerID, i int) error {
		lg, err := e.logits(xs[i])
		if err != nil {
			return err
		}
		if argmaxInt(lg) == labels[i] {
			counts[workerID]++
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	total := 0
	for _, c := range counts {
		total += c
	}
	return float64(total) / float64(len(xs)), nil
}

// predictLabels は全サンプルに対するアンサンブルの予測クラスを返す(蒸留の疑似ラベル)。
func (e *ensemble) predictLabels(xs bitsx.Matrices, p int) ([]int, error) {
	out := make([]int, len(xs))
	err := parallel.For(len(xs), p, func(workerID, i int) error {
		lg, err := e.logits(xs[i])
		if err != nil {
			return err
		}
		out[i] = argmaxInt(lg)
		return nil
	})
	return out, err
}

// voteCodes は、各サンプルについて教師モデルの最終活性をビットごとに多数決した符号を返す。
//
// 最終活性のビット j は「プロトタイプのビット j に合わせよ」という同じ意味を持つため、
// モデルをまたいで比較できる(重みと違って順列対称性の影響を受けない)。
// 多数決を取ることで、個々のモデルの誤りが打ち消され、教師の「自信のあるビット」が残る。
// 同数の場合はモデル数が偶数のときに起きるので、+1 側へ倒す
// (このとき票は正負同数であり、どちらに倒しても情報量は同じ)。
func (e *ensemble) voteCodes(xs bitsx.Matrices, p int) (bitsx.Matrices, error) {
	if len(e.models) == 0 {
		return nil, errors.New("モデルがありません")
	}
	out := make(bitsx.Matrices, len(xs))
	err := parallel.For(len(xs), p, func(workerID, i int) error {
		var tally []int
		rows, cols := 0, 0
		for _, m := range e.models {
			y, err := m.predict(xs[i])
			if err != nil {
				return err
			}
			if tally == nil {
				rows, cols = y.Rows(), y.Cols()
				tally = make([]int, rows*cols)
			}
			idx := 0
			if err := y.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
				word, err := y.Word(ctx.WordIndex)
				if err != nil {
					return err
				}
				return ctx.ScanBits(func(b, col, colT int) error {
					if word>>uint64(b)&1 == 1 {
						tally[idx]++
					} else {
						tally[idx]--
					}
					idx++
					return nil
				})
			}); err != nil {
				return err
			}
		}
		code, err := bitsx.NewZerosMatrix(rows, cols)
		if err != nil {
			return err
		}
		idx := 0
		if err := code.ScanRowsWord(nil, func(ctx bitsx.MatrixWordContext) error {
			var word uint64
			if err := ctx.ScanBits(func(b, col, colT int) error {
				if tally[idx] >= 0 {
					word |= 1 << uint64(b)
				}
				idx++
				return nil
			}); err != nil {
				return err
			}
			return code.SetWord(ctx.WordIndex, word)
		}); err != nil {
			return err
		}
		out[i] = code
		return nil
	})
	return out, err
}

// codeAgreement は、教師の多数決符号と正解プロトタイプの一致率を返す。
// 蒸留の目標がどれだけ「正解の符号」に近いかの診断。
func codeAgreement(codes bitsx.Matrices, labels []int, protos bitsx.Matrices) (float64, error) {
	var same, total int
	for i, code := range codes {
		hd, err := code.HammingDistance(protos[labels[i]])
		if err != nil {
			return 0, err
		}
		n := code.Rows() * code.Cols()
		same += n - hd
		total += n
	}
	return float64(same) / float64(total), nil
}
