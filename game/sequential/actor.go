package sequential

import (
	"fmt"
	"github.com/sw965/crow/game"
)

type PolicyFunc[S any, Ac comparable] func(S, []Ac) (game.Policy[Ac], error)

func UniformPolicyFunc[S any, Ac comparable](state S, legalActions []Ac) (game.Policy[Ac], error) {
	n := len(legalActions)
	if n == 0 {
		return nil, fmt.Errorf("legalActionsが空です: len(legalActions) > 0 であるべき")
	}

	p := 1.0 / float32(n)
	policy := game.Policy[Ac]{}
	for _, a := range legalActions {
		policy[a] = p
	}
	return policy, nil
}

type PolicyValueFunc[S any, Ac comparable] func(S, []Ac) (game.Policy[Ac], float32, error)

func UniformPolicyNoValueFunc[S any, Ac comparable](state S, legalActions []Ac) (game.Policy[Ac], float32, error) {
	policy, err := UniformPolicyFunc(state, legalActions)
	if err != nil {
		return nil, 0.0, err
	}
	return policy, 0.0, err
}

type ActorCritic[S any, Ac, Ag comparable] struct {
	Name            game.ActorCriticName
	PolicyValueFunc PolicyValueFunc[S, Ac]
	SelectFunc      game.SelectFunc[Ac, Ag]
}

func NewRandomActorCritic[S any, Ac, Ag comparable]() ActorCritic[S, Ac, Ag] {
	return ActorCritic[S, Ac, Ag]{
		Name:            "rand",
		PolicyValueFunc: UniformPolicyNoValueFunc[S, Ac],
		SelectFunc:      game.WeightedRandomSelectFunc[Ac, Ag],
	}
}

func (a ActorCritic[S, Ac, Ag]) Validate() error {
	if a.PolicyValueFunc == nil {
		return fmt.Errorf("PolicyValueFuncがnilです")
	}
	if a.SelectFunc == nil {
		return fmt.Errorf("SelectFuncがnilです")
	}
	return nil
}
