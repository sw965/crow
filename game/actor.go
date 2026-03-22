package game

import (
	"fmt"
	"github.com/sw965/omw/mathx"
	"github.com/sw965/omw/mathx/randx"
	"github.com/sw965/omw/slicesx"
	"maps"
	"math/rand/v2"
	"slices"
)

type Policy[Ac comparable] map[Ac]float32

func (p Policy[Ac]) ValidateForLegalActions(legalActions []Ac, checkUnique bool) error {
	if checkUnique {
		if !slicesx.IsUnique(legalActions) {
			return fmt.Errorf("legalActions contains duplicates")
		}
	}

	if len(legalActions) == 0 {
		return fmt.Errorf("legalActions must not be empty")
	}

	if len(p) != len(legalActions) {
		return fmt.Errorf("policy size (%d) does not match legal actions count (%d)", len(p), len(legalActions))
	}

	var sum float32
	for _, a := range legalActions {
		v, ok := p[a]
		if !ok {
			return fmt.Errorf("policy is missing probability for action: %v", a)
		}

		if v < 0 || mathx.IsNaN(v) || mathx.IsInf(v, 0) {
			return fmt.Errorf("invalid probability value %f for action: %v", v, a)
		}
		sum += v
	}

	if sum == 0 {
		return fmt.Errorf("sum of policy probabilities is zero")
	}
	return nil
}

type SelectFunc[Ac, Ag comparable] func(Policy[Ac], Ag, *rand.Rand) (Ac, error)

func MaxSelectFunc[Ac, Ag comparable](policy Policy[Ac], agent Ag, rng *rand.Rand) (Ac, error) {
	keys := slices.Collect(maps.Keys(policy))
	max := policy[keys[0]]
	// capの確保をする。
	actions := []Ac{keys[0]}

	for _, k := range keys[1:] {
		v := policy[k]
		switch {
		case v > max:
			max = v
			actions = []Ac{k}
		case v == max:
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