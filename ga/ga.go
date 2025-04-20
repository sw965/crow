package ga

import (
	"fmt"
	"math/rand"

	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
)

func LinearRankingProb(n, rank int, s float32) float32 {
	m := 2.0 - s
	return 1.0/float32(n) * (m + (s-m)*(float32(n-rank)/float32(n-1)))
}

const epsilon = 0.0001

type Population[I any] []I

type FitnessY float32
type FitnessYs []FitnessY
type Fitness[I any] func(Population[I], int) FitnessY

func (f Fitness[I]) Argsort(pop Population[I]) []int {
	fitYs := make(FitnessYs, len(pop))
	for i := range pop {
		fitYs[i] = f(pop, i)
	}
	idxs := omwslices.Argsort(fitYs)
	return idxs
}

func (f Fitness[I]) SortPopulation(pop Population[I]) Population[I] {
	idxs := f.Argsort(pop)
	idxs = omwslices.Reverse(idxs)

	sorted := make(Population[I], len(idxs))
	for i, idx := range idxs {
		sorted[i] = pop[idx]
	}
	return sorted
}

type IndexSelector func(FitnessYs, *rand.Rand) (int, error)

func RouletteIndexSelector(fitYs FitnessYs, r *rand.Rand) (int, error) {
	ws := make([]float32, len(fitYs))
	for i, fitY := range fitYs {
		ws[i] = float32(fitY)
	}
	idx := omwrand.IntByWeight(ws, r, epsilon)
	return idx, nil
}

func NewLinearRankingIndexSelector(s float32) IndexSelector {
	return func(fitYs FitnessYs, r *rand.Rand) (int, error) {
		argsorted := omwslices.Argsort(fitYs)
		argsorted = omwslices.Reverse(argsorted)

		n := len(fitYs)
		probs := make([]float32, len(fitYs))
		for i := range probs {
			probs[i] = LinearRankingProb(n, i, s)
		}
		idx := omwrand.IntByWeight(probs, r, epsilon)
		return argsorted[idx], nil
	}
}

type CrossOperator[I any] func(I, I, *rand.Rand) (I, I, error)

func UniformCrossOperator[I []T, T any](parent1, parent2 I, r *rand.Rand) (I, I, error) {
	n := len(parent1)
	if n != len(parent2) {
		return nil, nil, fmt.Errorf("個体の長さが一致しない")
	}

	child1 := make(I, n)
	child2 := make(I, n)
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

type MutationOperator[I any] func(I, *rand.Rand) I

func NewUniformMutationOperator[I []float32](min, max float32) MutationOperator[I] {
	return func(individual I, r *rand.Rand) I {
		mutated := make(I, len(individual))
		for i, gene := range individual {
			mutated[i] = gene + omwrand.Float64(min, max, r)
		}
		return mutated
	}
}

type CloneOperator[I any] func(I) I

type Engine[I any] struct {
	Fitness          Fitness[I]
	IndexSelector    IndexSelector
	CrossOperator    CrossOperator[I]
	MutationOperator MutationOperator[I]
	CloneOperator    CloneOperator[I]
	CrossPercent     float32
	MutationPercent  float32
	EliteNum         int
}

func (e *Engine[I]) Run(init Population[I], generation int, r *rand.Rand) (Population[I], error) {
	N := len(init)
	if e.EliteNum > N {
		return nil, fmt.Errorf("エリート数 > 集団数 になっている為、処理を続行出来ません。")
	}
	current := init

	for i := 0; i < generation; i++ {
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