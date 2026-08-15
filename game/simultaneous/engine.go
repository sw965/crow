package simultaneous

import (
	"errors"

	"github.com/sw965/crow/game"
)

type LegalActionsByAgent[Ac, Ag comparable] map[Ag][]Ac
type LegalActionsByAgentFunc[S any, Ac, Ag comparable] func(S) LegalActionsByAgent[Ac, Ag]

// JointAction は、全エージェントが同時に選んだ行動の組。
type JointAction[Ac, Ag comparable] map[Ag]Ac

type TransitionFunc[S any, Ac, Ag comparable] func(S, JointAction[Ac, Ag]) (S, error)
type EqualFunc[S any] func(S, S) bool

type Rule[S any, Ac, Ag comparable] struct {
	LegalActionsByAgentFunc LegalActionsByAgentFunc[S, Ac, Ag]
	TransitionFunc          TransitionFunc[S, Ac, Ag]
	EqualFunc               EqualFunc[S]
}

func (r Rule[S, Ac, Ag]) Validate() error {
	if r.LegalActionsByAgentFunc == nil {
		return errors.New("LegalActionsByAgentFuncがnilです")
	}
	if r.TransitionFunc == nil {
		return errors.New("TransitionFuncがnilです")
	}
	if r.EqualFunc == nil {
		return errors.New("EqualFuncがnilです")
	}
	return nil
}

type Engine[S any, Ac, Ag comparable] struct {
	Rule                   Rule[S, Ac, Ag]
	RankByAgentFunc        game.RankByAgentFunc[S, Ag]
	ResultScoreByAgentFunc game.ResultScoreByAgentFunc[Ag]
	Agents                 []Ag
	// MaxSteps はプレイアウト1回あたりの手数の上限。
	// 状態が循環し得るゲームでプレイアウトが終了しなくなるのを防ぐ。
	// 0の場合は無制限。上限に達した場合、Playouts / RecordPlayouts はエラーを返す。
	MaxSteps int
}

func (e Engine[S, Ac, Ag]) Validate() error {
	if err := e.Rule.Validate(); err != nil {
		return err
	}

	if e.RankByAgentFunc == nil {
		return errors.New("RankByAgentFuncがnilです")
	}

	if e.ResultScoreByAgentFunc == nil {
		return errors.New("ResultScoreByAgentFuncがnilです")
	}

	if len(e.Agents) == 0 {
		return errors.New("agentsが空です: 1体以上のエージェントが必要")
	}
	return nil
}

func (e Engine[S, Ac, Ag]) IsTerminal(state S) (bool, error) {
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
