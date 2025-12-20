package tictactoe_test

import (
	"errors"
	game "github.com/sw965/crow/game/sequential"
	ttt "github.com/sw965/crow/game/sequential/tictactoe"
	"maps"
	"slices"
	"testing"
)

func TestMarkOpposite(t *testing.T) {
	tests := []struct {
		name string
		mark ttt.Mark
		want ttt.Mark
	}{
		{
			name: "正常_丸入力",
			mark: ttt.Nought,
			want: ttt.Cross,
		},
		{
			name: "正常_バツ入力",
			mark: ttt.Cross,
			want: ttt.Nought,
		},
		{
			name: "準正常_空入力",
			mark: ttt.EmptyMark,
			want: ttt.EmptyMark,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Helper()
			got := tc.mark.Opposite()
			if got != tc.want {
				t.Errorf("want: %d, got: %d", got, tc.want)
			}
		})
	}
}

func TestBoardIsFull(t *testing.T) {
	tests := []struct {
		name  string
		board ttt.Board
		want  bool
	}{
		{
			name: "正常_NotFull",
			board: ttt.Board{
				{ttt.Nought, ttt.Cross, ttt.Nought},
				{ttt.Cross, ttt.EmptyMark, ttt.Cross},
				{ttt.Nought, ttt.Cross, ttt.Nought},
			},
			want: false,
		},
		{
			name: "正常_Full",
			board: ttt.Board{
				{ttt.Cross, ttt.Nought, ttt.Cross},
				{ttt.Nought, ttt.Cross, ttt.Nought},
				{ttt.Cross, ttt.Nought, ttt.Cross},
			},
			want: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Helper()
			got := tc.board.IsFull()
			if got != tc.want {
				t.Errorf("want: %t, got: %t", tc.want, got)
			}
		})
	}
}

func TestNewInitState(t *testing.T) {
	got := ttt.NewInitState()

	// ttt.EmptyMark が 0 じゃなくなった時に、察知出来るように、マジックナンバー(0)で書く
	wantBoard := ttt.Board{
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
	}
	// ttt.Nought が 1 じゃなくなった時に、察知できるように、ttt.Nougthではなく、1を代入する
	wantTurn := ttt.Mark(1)

	if got.Board != wantBoard {
		t.Errorf("wantBoard: %v, got.Baord: %v", got.Board, wantBoard)
	}

	if got.Turn != wantTurn {
		t.Errorf("wantTurn: %v, got.Turn: %v", got.Turn, wantTurn)
	}
}

func TestLegalMoves(t *testing.T) {
	tests := []struct {
		name  string
		state ttt.State
		want  []ttt.Move
	}{
		{
			name:  "正常_初期局面",
			state: ttt.NewInitState(),
			want: []ttt.Move{
				{Row: 0, Col: 0, Mark: ttt.Nought}, {Row: 0, Col: 1, Mark: ttt.Nought}, {Row: 0, Col: 2, Mark: ttt.Nought},
				{Row: 1, Col: 0, Mark: ttt.Nought}, {Row: 1, Col: 1, Mark: ttt.Nought}, {Row: 1, Col: 2, Mark: ttt.Nought},
				{Row: 2, Col: 0, Mark: ttt.Nought}, {Row: 2, Col: 1, Mark: ttt.Nought}, {Row: 2, Col: 2, Mark: ttt.Nought},
			},
		},
		{
			name: "正常_真ん中だけ空",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Cross, ttt.Nought, ttt.Cross},
					{ttt.Nought, ttt.EmptyMark, ttt.Nought},
					{ttt.Cross, ttt.Nought, ttt.Cross},
				},
				Turn: ttt.Cross,
			},
			want: []ttt.Move{{Row: 1, Col: 1, Mark: ttt.Cross}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ttt.LegalMoves(tc.state)
			if !slices.Equal(got, tc.want) {
				t.Errorf("want: %v, got: %v", tc.want, got)
			}
		})
	}
}

func TestMoveFunc(t *testing.T) {
	tests := []struct {
		name      string
		state     ttt.State
		move      ttt.Move
		want      ttt.State
		wantErr   bool
		wantErrIs error
	}{
		{
			name:  "正常_境界値_(0,0)",
			state: ttt.NewInitState(),
			move:  ttt.Move{Mark: ttt.Nought, Row: 0, Col: 0},
			want: ttt.State{
				Board: ttt.Board{
					{ttt.Nought, ttt.EmptyMark, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
				},
				Turn: ttt.Cross,
			},
		},
		{
			name:  "正常_境界値_(0,2)",
			state: ttt.NewInitState(),
			move:  ttt.Move{Mark: ttt.Nought, Row: 0, Col: 2},
			want: ttt.State{
				Board: ttt.Board{
					{ttt.EmptyMark, ttt.EmptyMark, ttt.Nought},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
				},
				Turn: ttt.Cross,
			},
		},
		{
			name:  "正常_境界値_(2,0)",
			state: ttt.NewInitState(),
			move:  ttt.Move{Mark: ttt.Nought, Row: 2, Col: 0},
			want: ttt.State{
				Board: ttt.Board{
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
					{ttt.Nought, ttt.EmptyMark, ttt.EmptyMark},
				},
				Turn: ttt.Cross,
			},
		},
		{
			name:  "正常_境界値_(2,2)",
			state: ttt.NewInitState(),
			move:  ttt.Move{Mark: ttt.Nought, Row: 2, Col: 2},
			want: ttt.State{
				Board: ttt.Board{
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.Nought},
				},
				Turn: ttt.Cross,
			},
		},
		// Rowの境界値テストの異常系
		{
			name:      "異常_境界値_Row下限越え(-1,0)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: -1, Col: 0},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		{
			name:      "異常_境界値_Row下限越え_Col最大(-1,2)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: -1, Col: 2},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		{
			name:      "異常_境界値_Row上限越え(3,0)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: 3, Col: 0},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		{
			name:      "異常_境界値_Row上限越え_Col最大(3,2)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: 3, Col: 2},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		// Colの境界値テストの異常系
		{
			name:      "異常_境界値_(0,-1)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: 0, Col: -1},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		{
			name:      "異常_境界値_(2,-1)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: 2, Col: -1},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		{
			name:      "異常_境界値_(0,3)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: 0, Col: 3},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		{
			name:      "異常_境界値_(2,3)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: 2, Col: 3},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		//RowとColの複合の境界値テストの異常系
		{
			name:      "異常_境界値_複合_(-1,-1)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: -1, Col: -1},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		{
			name:      "異常_境界値_複合_(-1,3)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: -1, Col: 3},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		{
			name:      "異常_境界値_複合_(3,-1)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: 3, Col: -1},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		{
			name:      "異常_境界値_複合_(3,3)",
			state:     ttt.NewInitState(),
			move:      ttt.Move{Mark: ttt.Nought, Row: 3, Col: 3},
			wantErr:   true,
			wantErrIs: ttt.ErrOutOfBounds,
		},
		{
			name: "異常_手番なし_TurnがEmptyMark",
			state: ttt.State{
				Board: ttt.Board{}, // 盤面の状態に関わらず、TurnがEmptyMarkならエラーになるべき
				Turn:  ttt.EmptyMark,
			},
			move: ttt.Move{
				Mark: ttt.Nought, // どのようなMoveであっても
				Row:  0,
				Col:  0,
			},
			wantErr:   true,
			wantErrIs: ttt.ErrNoActiveTurn,
		},
		{
			name:      "異常_手番違い_その1",
			state:     ttt.State{Turn: ttt.Nought},
			move:      ttt.Move{Mark: ttt.Cross},
			wantErr:   true,
			wantErrIs: ttt.ErrNotYourTurn,
		},
		{
			name:      "異常_手番違い_その2",
			state:     ttt.State{Turn: ttt.Cross},
			move:      ttt.Move{Mark: ttt.Nought},
			wantErr:   true,
			wantErrIs: ttt.ErrNotYourTurn,
		},
		{
			name: "異常_入力済み",
			state: ttt.State{
				// (2, 2) の地点が既にバツが入力されている
				Board: ttt.Board{
					{ttt.Cross, ttt.Nought, ttt.EmptyMark},
					{ttt.Nought, ttt.Nought, ttt.EmptyMark},
					{ttt.Cross, ttt.EmptyMark, ttt.Cross},
				},
				Turn: ttt.Nought,
			},

			// (2, 2) の地点に丸を入力
			move: ttt.Move{
				Mark: ttt.Nought,
				Row:  2,
				Col:  2,
			},

			wantErr:   true,
			wantErrIs: ttt.ErrCellOccupied,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Helper()
			got, err := ttt.MoveFunc(tc.state, tc.move)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("エラーを期待したが、nilが返された")
				}

				if !errors.Is(err, tc.wantErrIs) {
					t.Errorf("期待されるエラー型が埋め込まれていません。want: %v, got: %v", tc.wantErrIs, errors.Unwrap(err))
				}
				return
			}

			if err != nil {
				t.Fatalf("予期せぬエラーが発生した: %v", err)
			}

			if got != tc.want {
				t.Errorf("want: %+v, got: %+v", got, tc.want)
			}
		})
	}
}

func TestRankByAgentFunc(t *testing.T) {
	tests := []struct {
		name    string
		state   ttt.State
		want    game.RankByAgent[ttt.Mark]
		wantErr bool
	}{
		{
			name: "勝利_横ライン_上段_丸",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Nought, ttt.Nought, ttt.Nought},
					{ttt.Cross, ttt.Cross, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
				},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Nought: 1, ttt.Cross: 2},
		},
		{
			name: "勝利_横ライン_中段_バツ",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Nought, ttt.Nought, ttt.EmptyMark},
					{ttt.Cross, ttt.Cross, ttt.Cross},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
				},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Cross: 1, ttt.Nought: 2},
		},
		{
			name: "勝利_横ライン_下段_丸",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Cross, ttt.Cross, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
					{ttt.Nought, ttt.Nought, ttt.Nought},
				},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Nought: 1, ttt.Cross: 2},
		},
		{
			name: "勝利_縦ライン_左列_バツ",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Cross, ttt.Nought, ttt.EmptyMark},
					{ttt.Cross, ttt.Nought, ttt.EmptyMark},
					{ttt.Cross, ttt.EmptyMark, ttt.EmptyMark},
				},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Cross: 1, ttt.Nought: 2},
		},
		{
			name: "勝利_縦ライン_中列_丸",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Cross, ttt.Nought, ttt.EmptyMark},
					{ttt.Cross, ttt.Nought, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.Nought, ttt.Cross},
				},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Nought: 1, ttt.Cross: 2},
		},
		{
			name: "勝利_縦ライン_右列_バツ",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Nought, ttt.EmptyMark, ttt.Cross},
					{ttt.Nought, ttt.EmptyMark, ttt.Cross},
					{ttt.EmptyMark, ttt.Nought, ttt.Cross},
				},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Cross: 1, ttt.Nought: 2},
		},
		{
			name: "勝利_斜め_左上から右下_丸",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Nought, ttt.Cross, ttt.EmptyMark},
					{ttt.Cross, ttt.Nought, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.Cross, ttt.Nought},
				},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Nought: 1, ttt.Cross: 2},
		},
		{
			name: "勝利_斜め_右上から左下_バツ",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Nought, ttt.Nought, ttt.Cross},
					{ttt.Nought, ttt.Cross, ttt.EmptyMark},
					{ttt.Cross, ttt.EmptyMark, ttt.EmptyMark},
				},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Cross: 1, ttt.Nought: 2},
		},
		{
			name: "引き分け",
			state: ttt.State{
				// 全部埋まったが揃ってない
				Board: ttt.Board{
					{ttt.Nought, ttt.Cross, ttt.Nought},
					{ttt.Cross, ttt.Nought, ttt.Cross},
					{ttt.Cross, ttt.Nought, ttt.Cross},
				},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Nought: 1, ttt.Cross: 1},
		},
		{
			name: "進行中",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Nought, ttt.Cross, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
					{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
				},
			},
			//ゲームが続いている時は、空 or nil を返すのが仕様
			want: game.RankByAgent[ttt.Mark]{},
		},
		{
			name: "最終手で勝利",
			state: ttt.State{
				// 斜めが揃っている
				Board: ttt.Board{
					{ttt.Nought, ttt.Cross, ttt.Nought},
					{ttt.Cross, ttt.Nought, ttt.Cross},
					{ttt.Nought, ttt.Cross, ttt.Nought},
				},
			},
			// ここで IsFull() が true でも、勝利判定が優先されるべき
			want: game.RankByAgent[ttt.Mark]{ttt.Nought: 1, ttt.Cross: 2},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ttt.RankByAgentFunc(tc.state)
			if (err != nil) != tc.wantErr {
				t.Errorf("error = %v, wantErr %v", err, tc.wantErr)
				return
			}

			if !maps.Equal(got, tc.want) {
				t.Errorf("want:%v, got:%v", tc.want, got)
			}
		})
	}
}