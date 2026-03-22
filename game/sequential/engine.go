package sequential

import (
	"fmt"
	"github.com/sw965/crow/game"
)

type LegalActionsFunc[S any, Ac comparable] func(S) []Ac
type ActionFunc[S any, Ac comparable] func(S, Ac) (S, error)
type EqualFunc[S any] func(S, S) bool
type CurrentAgentFunc[S any, Ag comparable] func(S) Ag

type Logic[S any, Ac, Ag comparable] struct {
	LegalActionsFunc LegalActionsFunc[S, Ac]
	ActionFunc       ActionFunc[S, Ac]
	EqualFunc        EqualFunc[S]
	CurrentAgentFunc CurrentAgentFunc[S, Ag]
}

func (l Logic[S, Ac, Ag]) Validate() error {
	if l.LegalActionsFunc == nil {
		return fmt.Errorf("LegalActionsFunc must not be nil")
	}
	if l.ActionFunc == nil {
		return fmt.Errorf("ActionFunc must not be nil")
	}
	if l.EqualFunc == nil {
		return fmt.Errorf("EqualFunc must not be nil")
	}
	if l.CurrentAgentFunc == nil {
		return fmt.Errorf("CurrentAgentFunc must not be nil")
	}
	return nil
}

type Engine[S any, Ac, Ag comparable] struct {
	Logic                  Logic[S, Ac, Ag]
	RankByAgentFunc        game.RankByAgentFunc[S, Ag]
	ResultScoreByAgentFunc game.ResultScoreByAgentFunc[Ag]
	Agents                 []Ag
}

func (e Engine[S, Ac, Ag]) Validate() error {
	if err := e.Logic.Validate(); err != nil {
		return err
	}

	if e.RankByAgentFunc == nil {
		return fmt.Errorf("RankByAgentFunc must not be nil")
	}

	if e.ResultScoreByAgentFunc == nil {
		return fmt.Errorf("ResultScoreByAgentFunc must not be nil")
	}

	if len(e.Agents) == 0 {
		return fmt.Errorf("agents list must not be empty")
	}
	return nil
}

func (e Engine[S, Ac, Ag]) IsEnd(state S) (bool, error) {
	rankByAgent, err := e.RankByAgentFunc(state)
	return len(rankByAgent) != 0, err
}

func (e Engine[S, Ac, Ag]) EvaluateResultScoreByAgent(state S) (game.ResultScoreByAgent[Ag], error) {
	rankByAgent, err := e.RankByAgentFunc(state)
	if err != nil {
		return nil, err
	}
	return e.ResultScoreByAgentFunc(rankByAgent)
}

func (e *Engine[S, Ac, Ag]) SetStandardResultScoreByAgentFunc() {
	e.ResultScoreByAgentFunc = game.StandardResultScoreByAgentFunc[Ag]
}
