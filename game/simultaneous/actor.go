package simultaneous

import (
	"fmt"
	"github.com/sw965/crow/game"
)

type PolicyByAgent[M, A comparable] map[A]game.Policy[M]
type ValueByAgent[A comparable] map[A]float32

type PolicyValueFunc[S any, M, A comparable] func(S, LegalMovesByAgent[M, A]) (PolicyByAgent[M, A], ValueByAgent[A], error)

func UniformPolicyNoValueFunc[S any, M, A comparable](state S, legalMovesByAgent LegalMovesByAgent[M, A]) (PolicyByAgent[M, A], ValueByAgent[A], error) {
	jp := PolicyByAgent[M, A]{}
	jv := ValueByAgent[A]{}

	for agent, moves := range legalMovesByAgent {
		n := len(moves)
		if n == 0 {
			return nil, nil, fmt.Errorf("agent %v has no legal moves", agent)
		}
		p := 1.0 / float32(n)
		policy := game.Policy[M]{}
		for _, m := range moves {
			policy[m] = p
		}
		jp[agent] = policy
		jv[agent] = 0.0
	}
	return jp, jv, nil
}

type ActorCritic[S any, M, A comparable] struct {
	Name            game.ActorCriticName
	PolicyValueFunc PolicyValueFunc[S, M, A]
	SelectFunc      game.SelectFunc[M, A]
}

func NewRandomActorCritic[S any, M, A comparable]() ActorCritic[S, M, A] {
	return ActorCritic[S, M, A]{
		Name:            "rand",
		PolicyValueFunc: UniformPolicyNoValueFunc[S, M, A],
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
