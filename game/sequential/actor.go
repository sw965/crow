package sequential

import (
	"fmt"
	"github.com/sw965/crow/game"
)

type PolicyFunc[S any, M comparable] func(S, []M) (game.Policy[M], error)

func UniformPolicyFunc[S any, M comparable](state S, legalMoves []M) (game.Policy[M], error) {
	n := len(legalMoves)
	if n == 0 {
		return nil, fmt.Errorf("後でエラーメッセージを書く")
	}

	p := 1.0 / float32(n)
	policy := game.Policy[M]{}
	for _, a := range legalMoves {
		policy[a] = p
	}
	return policy, nil
}

type PolicyValueFunc[S any, M comparable] func(S, []M) (game.Policy[M], float32, error)

func UniformPolicyNoValueFunc[S any, M comparable](state S, legalMoves []M) (game.Policy[M], float32, error) {
	policy, err := UniformPolicyFunc(state, legalMoves)
	if err != nil {
		return nil, 0.0, err
	}
	return policy, 0.0, err
}

type ActorCritic[S any, M, A comparable] struct {
	Name            game.ActorCriticName
	PolicyValueFunc PolicyValueFunc[S, M]
	SelectFunc      game.SelectFunc[M, A]
}

func NewRandomActorCritic[S any, M, A comparable]() ActorCritic[S, M, A] {
	return ActorCritic[S, M, A]{
		Name:            "rand",
		PolicyValueFunc: UniformPolicyNoValueFunc[S, M],
		SelectFunc:      game.WeightedRandomSelectFunc[M, A],
	}
}

func (a ActorCritic[S, M, A]) Validate() error {
	if a.PolicyValueFunc == nil {
		return fmt.Errorf("PolicyValueFunc must not be nil")
	}
	if a.SelectFunc == nil {
		return fmt.Errorf("SelectFunc must not be nil")
	}
	return nil
}
