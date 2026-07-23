// Package ttt は、テスト用の題材としての三目並べ(Tic-Tac-Toe)を提供する。
// crow内の各パッケージ（game/sequential, mcts/puct 等）のテストから、
// 逐次手番ゲームの実例として使う事を目的としており、公開APIではない。
package ttt

import (
	"fmt"

	"github.com/sw965/crow/game"
	"github.com/sw965/crow/game/sequential"
)

// Mark は、盤面のマスの状態、及びエージェント(手番)を表す。
type Mark int

const (
	EmptyMark Mark = iota
	Nought         // 先手(〇)ではなく、単に一方のプレイヤー
	Cross
)

func (m Mark) String() string {
	switch m {
	case Nought:
		return "〇"
	case Cross:
		return "×"
	default:
		return "-"
	}
}

// opponent は、相手のマークを返す。
func opponent(m Mark) Mark {
	switch m {
	case Nought:
		return Cross
	case Cross:
		return Nought
	default:
		return EmptyMark
	}
}

type Board [3][3]Mark

// winner は、3つ揃っているマークを返す。揃っていない場合はEmptyMarkを返す。
func (b Board) winner() Mark {
	lines := [8][3][2]int{
		// 横
		{{0, 0}, {0, 1}, {0, 2}},
		{{1, 0}, {1, 1}, {1, 2}},
		{{2, 0}, {2, 1}, {2, 2}},
		// 縦
		{{0, 0}, {1, 0}, {2, 0}},
		{{0, 1}, {1, 1}, {2, 1}},
		{{0, 2}, {1, 2}, {2, 2}},
		// 斜め
		{{0, 0}, {1, 1}, {2, 2}},
		{{0, 2}, {1, 1}, {2, 0}},
	}

	for _, line := range lines {
		first := b[line[0][0]][line[0][1]]
		if first == EmptyMark {
			continue
		}
		if b[line[1][0]][line[1][1]] == first && b[line[2][0]][line[2][1]] == first {
			return first
		}
	}
	return EmptyMark
}

// isFull は、盤面が全て埋まっているかを返す。
func (b Board) isFull() bool {
	for _, row := range b {
		for _, mark := range row {
			if mark == EmptyMark {
				return false
			}
		}
	}
	return true
}

// Action は、マークを置くマスの位置。
type Action struct {
	Row int
	Col int
}

type State struct {
	Board Board
	Turn  Mark
}

// NewInitialState は、開始局面(Crossが先手)を返す。
func NewInitialState() State {
	return State{Turn: Cross}
}

func legalActions(s State) []Action {
	// 決着が付いている場合、合法手はない
	if s.Board.winner() != EmptyMark {
		return nil
	}

	actions := make([]Action, 0, 9)
	for r := range 3 {
		for c := range 3 {
			if s.Board[r][c] == EmptyMark {
				actions = append(actions, Action{Row: r, Col: c})
			}
		}
	}
	return actions
}

func transition(s State, a Action) (State, error) {
	if a.Row < 0 || a.Row > 2 || a.Col < 0 || a.Col > 2 {
		return State{}, fmt.Errorf("actionが範囲外: action = %v: RowとColは 0 <= v <= 2 であるべき", a)
	}

	if s.Board[a.Row][a.Col] != EmptyMark {
		return State{}, fmt.Errorf("既にマークが置かれています: action = %v, mark = %v", a, s.Board[a.Row][a.Col])
	}

	if s.Turn != Nought && s.Turn != Cross {
		return State{}, fmt.Errorf("Turnが不正: turn = %v: NoughtまたはCrossであるべき", s.Turn)
	}

	next := s
	next.Board[a.Row][a.Col] = s.Turn
	next.Turn = opponent(s.Turn)
	return next, nil
}

func rankByAgent(s State) (game.RankByAgent[Mark], error) {
	w := s.Board.winner()
	if w != EmptyMark {
		return game.RankByAgent[Mark]{w: 1, opponent(w): 2}, nil
	}

	// 引き分け
	if s.Board.isFull() {
		return game.RankByAgent[Mark]{Nought: 1, Cross: 1}, nil
	}

	// ゲームが終了していない場合は空を返す
	return game.RankByAgent[Mark]{}, nil
}

// NewEngine は、三目並べのゲームエンジンを返す。
func NewEngine() sequential.Engine[State, Action, Mark] {
	e := sequential.Engine[State, Action, Mark]{
		Logic: sequential.Logic[State, Action, Mark]{
			LegalActionsFunc: legalActions,
			TransitionFunc:   transition,
			EqualFunc:        func(s1, s2 State) bool { return s1 == s2 },
			CurrentAgentFunc: func(s State) Mark { return s.Turn },
		},
		RankByAgentFunc: rankByAgent,
		Agents:          []Mark{Cross, Nought},
	}
	e.SetStandardResultScoreByAgentFunc()
	return e
}
