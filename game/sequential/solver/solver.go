package solver

import (
	"math/rand"
	omwmath "github.com/sw965/omw/math"
	omwrand "github.com/sw965/omw/math/rand"
	omwslices "github.com/sw965/omw/slices"
	"golang.org/x/exp/maps"
)

type Policy[A comparable] map[A]float64
type PolicyProvider[S any, As ~[]A, A comparable] func(*S, As) Policy[A]

func UniformPolicyProvider[S any, As ~[]A, A comparable](state *S, legalActions As) Policy[A] {
	n := len(legalActions)
	p := 1.0 / float64(n)
	policy := Policy[A]{}
	for _, a := range legalActions {
		policy[a] = p
	}
	return policy
}

type Eval float64
type EvalPerAgent[G comparable] []Eval

type ActorCritic[S any] func(S) (Policy, Eval, error)
type Selector[A comparable] func(Policy[A]) A

func NewEpsilonGreedySelector[A comparable](e float64, r *rand.Rand) Selector[A] {
	return func(p Policy[A]) A {
		ks := maps.Keys(p)
		if e > r.Float64() {
			return omwrand.Choice(ks, r)
		}
		vs := maps.Values(p)
		idxs := omwslices.MaxIndices(vs)
		idx := omwrand.Choice(idxs, r)
		return ks[idx]
	}
}

func NewMaxSelector[A comparable](r *rand.Rand) Selector[A] {
	return NewEpsilonGreedySelector[A](0.0, r)
}

func NewThresholdWeightedSelector[A comparable](t float64, r *rand.Rand) Selector[A] {
	return func(policy Policy[A]) A {
		max := omwmath.Max(maps.Values(policy)...)
		threshold := max * t
		n := len(policy)
		options := make([]A, 0, n)
		weights := make([]float64, 0, n)
		for action, p := range policy {
			if p >= threshold {
				options = append(options, action)
				weights = append(weights, p)
			}
		}
		idx := omwrand.IntByWeight(weights, r)
		return options[idx]
	}
}

type Solver[A comparable] struct {
	ActorCritic ActorCritic[A]
	Selector Selector[A]
}

type PerAgent[A, G comparable] map[G]*Solver[A]