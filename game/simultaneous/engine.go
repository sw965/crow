package simultaneous

import (
	"fmt"
	"github.com/sw965/crow/game"
)

type LegalActionsByAgent[Ac, Ag comparable] map[Ag][]Ac
type LegalActionsByAgentFunc[S any, Ac, Ag comparable] func(S) LegalActionsByAgent[Ac, Ag]
type TransitionFunc[S any, Ac, Ag comparable] func(S, map[Ag]Ac) (S, error)
type EqualFunc[S any] func(S, S) bool

type Logic[S any, Ac, Ag comparable] struct {
	LegalActionsByAgentFunc LegalActionsByAgentFunc[S, Ac, Ag]
	TransitionFunc              TransitionFunc[S, Ac, Ag]
	EqualFunc             EqualFunc[S]
}

func (l Logic[S, Ac, Ag]) Validate() error {
	if l.LegalActionsByAgentFunc == nil {
		return fmt.Errorf("LegalActionsByAgentFunc must not be nil")
	}
	if l.TransitionFunc == nil {
		return fmt.Errorf("TransitionFunc must not be nil")
	}
	if l.EqualFunc == nil {
		return fmt.Errorf("EqualFunc must not be nil")
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