package ga

import (
	"fmt"
	"math/rand"
	"golang.org/x/exp/slices"
	omwrand "github.com/sw965/omw/math/rand"
)

type Individual[T any] []T

type Population[T any] []Individual[T]

type Fitness[T any] func(Population[T], int) float64

type Selector[T any] func(Population[T], Fitness[T], *rand.Rand) (Individual[T], error)

func RouletteSelector[T any](p Population[T], fit Fitness[T], r *rand.Rand) (Individual[T], error) {
	ws := make([]float64, len(p))
	for i := range ws {
		y := fit(p, i)
		if y < 0.0 {
			return nil, fmt.Errorf("ルーレット選択を用いる場合、適応関数の出力値は0以上でなければならない")
		}
		ws[i] = y
	}
	idx := omwrand.IntByWeight(ws, r)
	return p[idx], nil
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
	Selector         Selector[T]
	CrossOperator    CrossOperator[T]
	MutationOperator MutationOperator[T]
	CrossPercent     float64
	MutationPercent  float64
}

func (e *Engine[T]) Run(init Population[T], generation int, r *rand.Rand) (Population[T], error) {
	N := len(init)
	current := init

	for i := 0; i < generation; i++ {
		next := make(Population[T], 0, N)

		for len(next) < N {
			t := r.Float64()

			if t < e.CrossPercent {
				parent1, err := e.Selector(current, e.Fitness, r)
				if err != nil {
					return nil, err
				}

				parent2, err := e.Selector(current, e.Fitness, r)
				if err != nil {
					return nil, err
				}
				child1, child2, err := e.CrossOperator(parent1, parent2, r)
				if err != nil {
					return nil, err
				}

				next = append(next, child1)
				if len(next) < N {
					next = append(next, child2)
				}
			} else if t < e.CrossPercent+e.MutationPercent {
				parent, err := e.Selector(current, e.Fitness, r)
				if err != nil {
					return nil, err
				}
				mutated := e.MutationOperator(parent, r)
				next = append(next, mutated)
			} else {
				parent, err := e.Selector(current, e.Fitness, r)
				if err != nil {
					return nil, err
				}
				clone := slices.Clone(parent)
				next = append(next, clone)
			}
		}
		current = next
	}
	return current, nil
}