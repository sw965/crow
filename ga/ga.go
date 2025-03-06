package ga

import (
	"fmt"
	"math/rand"
	"golang.org/x/exp/slices"
	omwslices "github.com/sw965/omw/slices"
	omwrand "github.com/sw965/omw/math/rand"
)

func LinearRankingProb(n, rank int, s float64) float64 {
	m := 2.0 - s
	return 1.0 / float64(n) * (m  + (s - m) * (float64(n - rank) /float64(n - 1)))
}

type Individual[T any] []T
type Population[T any] []Individual[T]

type EvalY float64
type EvalYs []EvalY
type Evaluator[T any] func(Population[T], int) EvalY

type IndexSelector func(EvalYs, *rand.Rand) (int, error)

func RouletteIndexSelector(evalYs EvalYs, r *rand.Rand) (int, error) {
	ws := make([]float64, len(evalYs))
	for i, evalY := range evalYs {
		ws[i] = float64(evalY)
	}
	idx := omwrand.IntByWeight(ws, r)
	return idx, nil
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
	Evaluator        Evaluator[T]
	IndexSelector    IndexSelector
	CrossOperator    CrossOperator[T]
	MutationOperator MutationOperator[T]
	CrossPercent     float64
	MutationPercent  float64
}
/*
	omwslices.Argsortを使えばいいかも
	実装はこんな感じになってるよ。

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
	omwslices.Reverseを使う事も出来るよ。
*/

func (e *Engine[T]) Run(init Population[T], generation int, r *rand.Rand) (Population[T], error) {
	N := len(init)
	current := init

	for i := 0; i < generation; i++ {
		next := make(Population[T], 0, N)
		evalYs := make(EvalYs, N)
		for i := range current {
			evalYs[i] = e.Evaluator(current, i)
		}

		for len(next) < N {
			t := r.Float64()
			if t < e.CrossPercent {
				parent1Idx, err := e.IndexSelector(evalYs, r)
				if err != nil {
					return nil, err
				}

				parent2Idx, err := e.IndexSelector(evalYs, r)
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
				idx, err := e.IndexSelector(evalYs, r)
				if err != nil {
					return nil, err
				}
				parent := current[idx]
				mutated := e.MutationOperator(parent, r)
				next = append(next, mutated)
			} else {
				idx, err := e.IndexSelector(evalYs, r)
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