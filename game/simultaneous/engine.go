package simultaneous

import (
	"fmt"
	"github.com/sw965/crow/game"
)

type LegalMovesByAgent[M, A comparable] map[A][]M
type LegalMovesByAgentFunc[S any, M, A comparable] func(S) LegalMovesByAgent[M, A]
type MoveFunc[S any, M, A comparable] func(S, map[A]M) (S, error)
type EqualFunc[S any] func(S, S) bool

type Logic[S any, M, A comparable] struct {
	LegalMovesByAgentFunc LegalMovesByAgentFunc[S, M, A]
	MoveFunc              MoveFunc[S, M, A]
	EqualFunc             EqualFunc[S]
}

func (l Logic[S, M, A]) Validate() error {
	if l.LegalMovesByAgentFunc == nil {
		return fmt.Errorf("LegalMovesFunc must not be nil")
	}
	if l.MoveFunc == nil {
		return fmt.Errorf("MoveFunc must not be nil")
	}
	if l.EqualFunc == nil {
		return fmt.Errorf("EqualFunc must not be nil")
	}
	return nil
}

type Engine[S any, M, A comparable] struct {
	Logic                  Logic[S, M, A]
	RankByAgentFunc        game.RankByAgentFunc[S, A]
	ResultScoreByAgentFunc game.ResultScoreByAgentFunc[A]
	Agents                 []A
}

func (e Engine[S, M, A]) Validate() error {
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

func (e Engine[S, M, A]) IsEnd(state S) (bool, error) {
	rankByAgent, err := e.RankByAgentFunc(state)
	return len(rankByAgent) != 0, err
}

func (e Engine[S, M, A]) EvaluateResultScoreByAgent(state S) (game.ResultScoreByAgent[A], error) {
	rankByAgent, err := e.RankByAgentFunc(state)
	if err != nil {
		return nil, err
	}
	return e.ResultScoreByAgentFunc(rankByAgent)
}

func (e *Engine[S, M, A]) SetStandardResultScoreByAgentFunc() {
	e.ResultScoreByAgentFunc = game.StandardResultScoreByAgentFunc[A]
}