package simultaneous

import (
	"errors"
	"fmt"

	"github.com/sw965/crow/game"
)

type PolicyByAgent[Ac, Ag comparable] map[Ag]game.Policy[Ac]
type ValueByAgent[Ag comparable] map[Ag]float32

type PolicyValueFunc[S any, Ac, Ag comparable] func(S, LegalActionsByAgent[Ac, Ag]) (PolicyByAgent[Ac, Ag], ValueByAgent[Ag], error)

func UniformPolicyNoValueFunc[S any, Ac, Ag comparable](state S, legalActionsByAgent LegalActionsByAgent[Ac, Ag]) (PolicyByAgent[Ac, Ag], ValueByAgent[Ag], error) {
	policyByAgent := PolicyByAgent[Ac, Ag]{}
	valueByAgent := ValueByAgent[Ag]{}

	for agent, actions := range legalActionsByAgent {
		n := len(actions)
		if n == 0 {
			return nil, nil, fmt.Errorf("エージェントに合法手がありません: agent = %v", agent)
		}
		p := 1.0 / float32(n)
		policy := game.Policy[Ac]{}
		for _, a := range actions {
			policy[a] = p
		}
		policyByAgent[agent] = policy
		valueByAgent[agent] = 0.0
	}
	return policyByAgent, valueByAgent, nil
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
		return errors.New("PolicyValueFuncがnilです")
	}
	if a.SelectFunc == nil {
		return errors.New("SelectFuncがnilです")
	}
	return nil
}
