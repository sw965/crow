package tictactoe

import (
	"fmt"
	"github.com/sw965/crow/game"
	"github.com/sw965/crow/game/sequential"
)

type Mark int

const (
	EmptyMark Mark = iota
	Nought
	Cross
)

func (m Mark) Opposite() Mark {
	if m == Nought {
		return Cross
	} else if m == Cross {
		return Nought
	}
	return m
}

const (
	Rows = 3
	Cols = 3
)

// Board represents the 3x3 Tic-Tac-Toe grid.
//
// Boardは3x3の三目並べの盤面を表します。
type Board [Rows][Cols]Mark

// IsFull checks if all cells on the board are occupied.
//
// IsFullは、盤面の全てのセルが埋まっているかを確認します。
func (b Board) IsFull() bool {
	for _, row := range b {
		for _, mark := range row {
			if mark == EmptyMark {
				return false
			}
		}
	}
	return true
}

// Move represents a player's action, specifying the mark and the position.
//
// Moveはプレイヤーの行動を表し、記号と配置する場所（行・列）を指定します。
type Move struct {
	Mark Mark
	Row  int
	Col  int
}

// State holds the current board situation and the active player.
//
// Stateは、現在の盤面状況と手番のプレイヤーを保持します。
type State struct {
	Board Board
	Turn  Mark
}

// NewInitState creates a new initial game state.
//
// NewInitStateは、ゲームの初期状態を作成します。
func NewInitState() State {
	return State{
		Board: Board{},
		Turn:  Nought,
	}
}

// LegalMoves returns all available moves for the current state.
//
// LegalMovesは、現在の状態から可能な全ての合法手を返します。
func LegalMoves(state State) []Move {
	moves := make([]Move, 0, Rows*Cols)
	for i, row := range state.Board {
		for j, mark := range row {
			if mark == EmptyMark {
				moves = append(moves, Move{Mark: state.Turn, Row: i, Col: j})
			}
		}
	}
	return moves
}

// MoveFunc applies a move to the current state and returns the next state.
//
// MoveFuncは、現在の状態に行動を適用し、次の状態を返します。
func MoveFunc(state State, move Move) (State, error) {
	if state.Turn == EmptyMark {
		return State{}, fmt.Errorf("")
	}

	if state.Turn != move.Mark {
		return State{}, fmt.Errorf("")
	}

	if move.Row < 0 || move.Row >= Rows || move.Col < 0 || move.Col >= Cols {
		return State{}, fmt.Errorf("")
	}

	if state.Board[move.Row][move.Col] != EmptyMark {
		return State{}, fmt.Errorf("")
	}

	next := state
	next.Board[move.Row][move.Col] = move.Mark

	var nextTurn Mark
	if state.Turn == Nought {
		nextTurn = Cross
	} else if state.Turn == Cross {
		nextTurn = Nought
	}
	next.Turn = nextTurn

	return next, nil
}

// NewLogic creates a new Logic instance for the Tic-Tac-Toe game.
//
// NewLogicは、三目並べゲームのための新しいLogicインスタンスを作成します。
func NewLogic() sequential.Logic[State, Move, Mark] {
	return sequential.Logic[State, Move, Mark]{
		LegalMovesFunc: LegalMoves,
		MoveFunc:       MoveFunc,
		EqualFunc: func(s1, s2 State) bool {
			return s1 == s2
		},
		CurrentAgentFunc: func(s State) Mark {
			return s.Turn
		},
	}
}

// RankByAgent determines the ranking of agents based on the current state.
// It checks for a win or a draw. If the game is ongoing, it returns an empty map.
//
// RankByAgentは、現在の状態に基づいてエージェントの順位を決定します。
// 勝利または引き分けを確認します。ゲームが継続中の場合は、空のマップを返します。
func RankByAgentFunc(state State) (game.RankByAgent[Mark], error) {
	// 勝敗が決まるライン(縦、横、斜め）
	lines := [][3][2]int{
		{{0, 0}, {0, 1}, {0, 2}}, // 上段の横ライン
		{{1, 0}, {1, 1}, {1, 2}}, // 中段の横ライン
		{{2, 0}, {2, 1}, {2, 2}}, // 下段の横ライン
		{{0, 0}, {1, 0}, {2, 0}}, // 左の縦ライン
		{{0, 1}, {1, 1}, {2, 1}}, // 真ん中の縦ライン
		{{0, 2}, {1, 2}, {2, 2}}, // 右の縦ライン
		{{0, 0}, {1, 1}, {2, 2}}, // 左上から右下へのライン
		{{0, 2}, {1, 1}, {2, 0}}, // 右上から左下へのライン
	}

	for _, line := range lines {
		m1 := state.Board[line[0][0]][line[0][1]]
		m2 := state.Board[line[1][0]][line[1][1]]
		m3 := state.Board[line[2][0]][line[2][1]]

		// m1 == m2 && m2 == m3 ならば、ライン上に並んだ全てのマークが同じであるという事
		// それに加えて、 m1 != EmptyMarkであれば、〇か×かのいずれかが揃ったことを意味する
		if m1 != EmptyMark && m1 == m2 && m2 == m3 {
			winner := m1
			loser := winner.Opposite()
			// 勝者が1位、敗者が2位
			return game.RankByAgent[Mark]{
				winner: 1,
				loser:  2,
			}, nil
		}
	}

	// 引き分け
	if state.Board.IsFull() {
		return game.RankByAgent[Mark]{
			Nought: 1,
			Cross:  1,
		}, nil
	}

	// ゲームが終了していない場合
	return game.RankByAgent[Mark]{}, nil
}

func NewEngine() sequential.Engine[State, Move, Mark] {
	engine := sequential.Engine[State, Move, Mark]{
		Logic:           NewLogic(),
		RankByAgentFunc: RankByAgentFunc,
		Agents:          []Mark{Nought, Cross},
	}
	engine.SetStandardResultScoreByAgentFunc()
	return engine
}
