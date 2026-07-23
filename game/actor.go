package game

import (
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/slicesx"
)

type Policy[Ac comparable] map[Ac]float32

func (p Policy[Ac]) ValidateForLegalActions(legalActions []Ac, checkUnique bool) error {
	if checkUnique {
		if !slicesx.IsUnique(legalActions) {
			return fmt.Errorf("legalActionsが重複しています")
		}
	}

	if len(legalActions) == 0 {
		return fmt.Errorf("legalActionsが空です: len(legalActions) > 0 であるべき")
	}

	if len(p) != len(legalActions) {
		return fmt.Errorf("policyとlegalActionsの要素数が不一致: len(policy) = %d, len(legalActions) = %d", len(p), len(legalActions))
	}

	var sum float32
	for _, a := range legalActions {
		v, ok := p[a]
		if !ok {
			return fmt.Errorf("policyにactionの確率が存在しません: action = %v", a)
		}

		if v < 0 || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("確率が不正(負/NaN/Inf): action = %v, p = %f", a, v)
		}
		sum += v
	}

	if sum == 0 {
		return fmt.Errorf("policyの確率の合計が0です: 合計は正であるべき")
	}
	return nil
}

type SelectFunc[Ac, Ag comparable] func(Policy[Ac], Ag, *rand.Rand) (Ac, error)

func MaxSelectFunc[Ac, Ag comparable](policy Policy[Ac], agent Ag, rng *rand.Rand) (Ac, error) {
	if len(policy) == 0 {
		var zero Ac
		return zero, fmt.Errorf("policyが空です: len(policy) > 0 であるべき")
	}

	keys := slices.Collect(maps.Keys(policy))
	maxP := policy[keys[0]]
	actions := []Ac{keys[0]}

	for _, k := range keys[1:] {
		v := policy[k]
		switch {
		case v > maxP:
			maxP = v
			actions = []Ac{k}
		case v == maxP:
			actions = append(actions, k)
		}
	}

	action, err := randx.Choice(actions, rng)
	if err != nil {
		var zero Ac
		return zero, err
	}
	return action, nil
}

func WeightedRandomSelectFunc[Ac, Ag comparable](policy Policy[Ac], agent Ag, rng *rand.Rand) (Ac, error) {
	n := len(policy)
	actions := make([]Ac, 0, n)
	ws := make([]float32, 0, n)
	for a, p := range policy {
		actions = append(actions, a)
		ws = append(ws, p)
	}

	idx, err := randx.IntByWeights(ws, rng)
	if err != nil {
		var zero Ac
		return zero, err
	}
	return actions[idx], nil
}

type ActorCriticName string
