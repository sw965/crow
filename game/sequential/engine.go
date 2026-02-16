package sequential

import (
	"fmt"
)

type LegalMovesFunc[S any, M comparable] func(S) []M
type MoveFunc[S any, M comparable] func(S, M) (S, error)
type EqualFunc[S any] func(S, S) bool
type CurrentAgentFunc[S any, A comparable] func(S) A

type Logic[S any, M, A comparable] struct {
	LegalMovesFunc   LegalMovesFunc[S, M]
	MoveFunc         MoveFunc[S, M]
	EqualFunc        EqualFunc[S]
	CurrentAgentFunc CurrentAgentFunc[S, A]
}

func (l Logic[S, M, A]) Validate() error {
	if l.LegalMovesFunc == nil {
		return fmt.Errorf("LegalMovesFunc must not be nil")
	}
	if l.MoveFunc == nil {
		return fmt.Errorf("MoveFunc must not be nil")
	}
	if l.EqualFunc == nil {
		return fmt.Errorf("EqualFunc must not be nil")
	}
	if l.CurrentAgentFunc == nil {
		return fmt.Errorf("CurrentAgentFunc must not be nil")
	}
	return nil
}

type Engine[S any, M, A comparable] struct {
	Logic                  Logic[S, M, A]
	RankByAgentFunc        RankByAgentFunc[S, A]
	ResultScoreByAgentFunc ResultScoreByAgentFunc[A]
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
