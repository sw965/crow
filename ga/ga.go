package ga

import (
	"fmt"
	"math/rand"

	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
)

// 線形ランキング選択で使う確率計算
func LinearRankingProb(n, rank int, s float64) float64 {
	m := 2.0 - s
	return 1.0/float64(n) * (m + (s-m)*(float64(n-rank)/float64(n-1)))
}

const epsilon = 0.0001

// Population は個体の集まり。個体の型は I で任意の型を許容する。
type Population[I any] []I

type FitnessY float64
type FitnessYs []FitnessY

// Fitness は、集団と個体のインデックスから適応度を計算する関数
type Fitness[I any] func(Population[I], int) FitnessY

// IndexSelector は、適応度リストと乱数生成器から個体のインデックスを選ぶ関数
type IndexSelector func(FitnessYs, *rand.Rand) (int, error)

// ルーレット選択の例
func RouletteIndexSelector(fitYs FitnessYs, r *rand.Rand) (int, error) {
	ws := make([]float64, len(fitYs))
	for i, fitY := range fitYs {
		ws[i] = float64(fitY)
	}
	idx := omwrand.IntByWeight(ws, r, epsilon)
	return idx, nil
}

// 線形ランキング選択の例。選択圧 s をパラメータとして受け取る。
func NewLinearRankingIndexSelector(s float64) IndexSelector {
	return func(fitYs FitnessYs, r *rand.Rand) (int, error) {
		argsorted := omwslices.Argsort(fitYs)
		argsorted = omwslices.Reverse(argsorted)

		n := len(fitYs)
		probs := make([]float64, len(fitYs))
		for i := range probs {
			probs[i] = LinearRankingProb(n, i, s)
		}
		idx := omwrand.IntByWeight(probs, r, epsilon)
		return argsorted[idx], nil
	}
}

// CrossOperator は、2つの個体から交叉を行い2つの子個体を生成するオペレーター
type CrossOperator[I any] func(I, I, *rand.Rand) (I, I, error)

// 例としての一様交叉。個体がスライスの場合に有効。
// ※個体がスライス以外の場合は、ユーザーが適切な交叉関数を実装する必要があります。
func UniformCrossOperator[T any](parent1, parent2 []T, r *rand.Rand) ([]T, []T, error) {
	n := len(parent1)
	if n != len(parent2) {
		return nil, nil, fmt.Errorf("個体の長さが一致しない")
	}

	child1 := make([]T, n)
	child2 := make([]T, n)
	for i := range parent1 {
		if r.Float64() < 0.5 {
			child1[i] = parent1[i]
			child2[i] = parent2[i]
		} else {
			child1[i] = parent2[i]
			child2[i] = parent1[i]
		}
	}
	return child1, child2, nil
}

// MutationOperator は、個体に突然変異を加えるオペレーター
type MutationOperator[I any] func(I, *rand.Rand) I

// CloneOperator は、個体をディープコピーするための関数。
// 個体がスライスの場合は、golang.org/x/exp/slices の Clone() などが利用できます。
type CloneOperator[I any] func(I) I

// Engine は、個体の型 I に対する遺伝的アルゴリズムのエンジン
type Engine[I any] struct {
	Fitness          Fitness[I]
	IndexSelector    IndexSelector
	CrossOperator    CrossOperator[I]
	MutationOperator MutationOperator[I]
	CloneOperator    CloneOperator[I]
	CrossPercent     float64 // 交叉が行われる確率
	MutationPercent  float64 // 突然変異が行われる確率
	EliteNum         int     // エリート個体の数
}

func (e *Engine[I]) Run(init Population[I], generation int, r *rand.Rand) (Population[I], error) {
	N := len(init)
	if e.EliteNum > N {
		return nil, fmt.Errorf("エリート数 > 集団数 になっている為、処理を続行出来ません。")
	}
	current := init

	for gen := 0; gen < generation; gen++ {
		next := make(Population[I], 0, N)
		fitYs := make(FitnessYs, N)
		for j := range current {
			fitYs[j] = e.Fitness(current, j)
		}

		// エリート戦略: 適応度の高い上位 EliteNum 個体をそのまま次世代に引き継ぐ
		idxs := omwslices.Argsort(fitYs)
		idxs = omwslices.Reverse(idxs)
		for j := 0; j < e.EliteNum && j < len(idxs); j++ {
			eliteIdx := idxs[j]
			elite := e.CloneOperator(current[eliteIdx])
			next = append(next, elite)
		}

		// 残りの個体を生成
		for len(next) < N {
			t := r.Float64()
			if t < e.CrossPercent {
				// 交叉を行う
				parent1Idx, err := e.IndexSelector(fitYs, r)
				if err != nil {
					return nil, err
				}
				parent2Idx, err := e.IndexSelector(fitYs, r)
				if err != nil {
					return nil, err
				}
				parent1, parent2 := current[parent1Idx], current[parent2Idx]
				child1, child2, err := e.CrossOperator(parent1, parent2, r)
				if err != nil {
					return nil, err
				}
				next = append(next, child1)
				if len(next) < N {
					next = append(next, child2)
				}
			} else if t < e.CrossPercent+e.MutationPercent {
				// 突然変異を行う
				idx, err := e.IndexSelector(fitYs, r)
				if err != nil {
					return nil, err
				}
				parent := current[idx]
				mutated := e.MutationOperator(parent, r)
				next = append(next, mutated)
			} else {
				// そのままコピー（クローン）する
				idx, err := e.IndexSelector(fitYs, r)
				if err != nil {
					return nil, err
				}
				parent := current[idx]
				clone := e.CloneOperator(parent)
				next = append(next, clone)
			}
		}
		current = next
	}
	return current, nil
}