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

type Policy[M comparable] map[M]float32

func (p Policy[M]) ValidateForLegalMoves(legalMoves []M, checkUnique bool) error {
	if checkUnique {
		if !slicesx.IsUnique(legalMoves) {
			return fmt.Errorf("legalMoves contains duplicates")
		}
	}

	if len(legalMoves) == 0 {
		return fmt.Errorf("legalMoves must not be empty")
	}

	if len(p) != len(legalMoves) {
		return fmt.Errorf("policy size (%d) does not match legal moves count (%d)", len(p), len(legalMoves))
	}

	var sum float32
	for _, m := range legalMoves {
		v, ok := p[m]
		if !ok {
			return fmt.Errorf("policy is missing probability for move: %v", m)
		}

		if v < 0 || mathx.IsNaN(v) || mathx.IsInf(v, 0) {
			return fmt.Errorf("invalid probability value %f for move: %v", v, m)
		}
		sum += v
	}

	if sum == 0 {
		return fmt.Errorf("sum of policy probabilities is zero")
	}
	return nil
}

type SelectFunc[M, A comparable] func(Policy[M], A, *rand.Rand) (M, error)

func MaxSelectFunc[M, A comparable](policy Policy[M], agent A, rng *rand.Rand) (M, error) {
	keys := slices.Collect(maps.Keys(policy))
	max := policy[keys[0]]
	// capの確保をする。
	moves := []M{keys[0]}

	for _, k := range keys[1:] {
		v := policy[k]
		switch {
		case v > max:
			max = v
			moves = []M{k}
		case v == max:
			moves = append(moves, k)
		}
	}

	move, err := randx.Choice(moves, rng)
	if err != nil {
		var zero M
		return zero, err
	}
	return move, nil
}

func WeightedRandomSelectFunc[M, A comparable](policy Policy[M], agent A, rng *rand.Rand) (M, error) {
	n := len(policy)
	moves := make([]M, 0, n)
	ws := make([]float32, 0, n)
	for m, p := range policy {
		moves = append(moves, m)
		ws = append(ws, p)
	}

	idx, err := randx.IntByWeights(ws, rng)
	if err != nil {
		var zero M
		return zero, err
	}
	return moves[idx], nil
}

type ActorCriticName string