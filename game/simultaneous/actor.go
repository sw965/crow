package simultaneous

import (
	"fmt"
	"github.com/sw965/crow/game"
)

type PolicyByAgent[Ac, Ag comparable] map[Ag]game.Policy[Ac]
type ValueByAgent[Ag comparable] map[Ag]float32

type PolicyValueFunc[S any, Ac, Ag comparable] func(S, LegalActionsByAgent[Ac, Ag]) (PolicyByAgent[Ac, Ag], ValueByAgent[Ag], error)

func UniformPolicyNoValueFunc[S any, Ac, Ag comparable](state S, legalActionsByAgent LegalActionsByAgent[Ac, Ag]) (PolicyByAgent[Ac, Ag], ValueByAgent[Ag], error) {
	jp := PolicyByAgent[Ac, Ag]{}
	jv := ValueByAgent[Ag]{}

	for agent, actions := range legalActionsByAgent {
		n := len(actions)
		if n == 0 {
			return nil, nil, fmt.Errorf("agent %v has no legal actions", agent)
		}
		p := 1.0 / float32(n)
		policy := game.Policy[Ac]{}
		for _, a := range actions {
			policy[a] = p
		}
		jp[agent] = policy
		jv[agent] = 0.0
	}
	return jp, jv, nil
}

type ActorCritic[S any, Ac, Ag comparable] struct {
	Name            game.ActorCriticName
	PolicyValueFunc PolicyValueFunc[S, Ac, Ag]
	SelectFunc      game.SelectFunc[Ac, Ag]
}

func NewRandomActorCritic[S any, Ac, Ag comparable]() ActorCritic[S, Ac, Ag] {
	return ActorCritic[S, Ac, Ag]{
		Name:            "rand",
		PolicyValueFunc: UniformPolicyNoValueFunc[S, Ac, Ag],
		SelectFunc:      game.WeightedRandomSelectFunc[Ac, Ag],
	}
}

func (a ActorCritic[S, Ac, Ag]) Validate() error {
	if a.PolicyValueFunc == nil {
		return fmt.Errorf("PolicyValueFunc must not be nil")
	}
	if a.SelectFunc == nil {
		return fmt.Errorf("SelectFunc must not be nil")
	}
	return nil
}
