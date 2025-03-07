package ga

import (
	"fmt"
	"math/rand"
	"golang.org/x/exp/slices"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
)

func LinearRankingProb(n, rank int, s float64) float64 {
	m := 2.0 - s
	return 1.0 / float64(n) * (m  + (s - m) * (float64(n - rank) /float64(n - 1)))
}

type Individual[T any] []T
type Population[T any] []Individual[T]

type FitnessY float64
type FitnessYs []FitnessY
type Fitness[T any] func(Population[T], int) FitnessY

type IndexSelector func(FitnessYs, *rand.Rand) (int, error)

func RouletteIndexSelector(fitYs FitnessYs, r *rand.Rand) (int, error) {
	ws := make([]float64, len(fitYs))
	for i, fitY := range fitYs {
		ws[i] = float64(fitY)
	}
	idx, err := omwrand.IntByWeight(ws, r)
	return idx, err
}

func NewLinearRankingIndexSelector(s float64) IndexSelector {
	return func(fitYs FitnessYs, r *rand.Rand) (int, error) {
		argsorted := omwslices.Argsort(fitYs)
		argsorted = omwslices.Reverse(argsorted)

		n := len(fitYs)
		probs := make([]float64, len(fitYs))
		for i := range probs {
			probs[i] = LinearRankingProb(n, i, s)
		}
		idx, err := omwrand.IntByWeight(probs, r)
		return idx, err
	}
}

type CrossOperator[T any] func(Individual[T], Individual[T], *rand.Rand) (Individual[T], Individual[T], error)

func UniformCrossOperator[T any](parent1, parent2 Individual[T], r *rand.Rand) (Individual[T], Individual[T], error) {
	n := len(parent1)
	if n != len(parent2) {
		return nil, nil, fmt.Errorf("個体の長さが一致しない")
	}

	child1 := make(Individual[T], n)
	child2 := make(Individual[T], n)

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

type MutationOperator[T any] func(Individual[T], *rand.Rand) Individual[T]

type Engine[T any] struct {
	Fitness          Fitness[T]
	IndexSelector    IndexSelector
	CrossOperator    CrossOperator[T]
	MutationOperator MutationOperator[T]
	CrossPercent     float64
	MutationPercent  float64
	EliteNum         int
}
/*
	omwslices.Argsortの実装はこんな感じになってるよ。

	func Argsort[S ~[]E, E constraints.Ordered](s S) []int {
    idxs := make([]int, len(s))
    for i := range s {
        idxs[i] = i
    }

    slices.SortFunc(idxs, func(idx1, idx2 int) bool {
		return s[idx1] < s[idx2]
    })

    return idxs
}
*/

func (e *Engine[T]) Run(init Population[T], generation int, r *rand.Rand) (Population[T], error) {
	N := len(init)
	current := init

	for i := 0; i < generation; i++ {
		next := make(Population[T], 0, N)
		fitYs := make(FitnessYs, N)
		for j := range current {
			fitYs[j] = e.Fitness(current, j)
		}

		// エリート戦略
		idxs := omwslices.Argsort(fitYs)
		idxs = omwslices.Reverse(idxs)
		for j := 0; j < e.EliteNum; j++ {
			eliteIdx := idxs[j]
			elite := slices.Clone(current[eliteIdx])
			next = append(next, elite)
		}

		for len(next) < N {
			t := r.Float64()
			if t < e.CrossPercent {
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
				idx, err := e.IndexSelector(fitYs, r)
				if err != nil {
					return nil, err
				}
				parent := current[idx]
				mutated := e.MutationOperator(parent, r)
				next = append(next, mutated)
			} else {
				idx, err := e.IndexSelector(fitYs, r)
				if err != nil {
					return nil, err
				}
				parent := current[idx]
				clone := slices.Clone(parent)
				next = append(next, clone)
			}
		}
		current = next
	}
	return current, nil
}