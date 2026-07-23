package ttt_test

import (
	"testing"

	"github.com/sw965/crow/game"
	"github.com/sw965/crow/internal/ttt"
)

func TestRankByAgent(t *testing.T) {
	engine := ttt.NewEngine()

	tests := []struct {
		name  string
		board ttt.Board
		want  game.RankByAgent[ttt.Mark]
	}{
		{
			name: "横一列で勝ち",
			board: ttt.Board{
				{ttt.Cross, ttt.Cross, ttt.Cross},
				{ttt.Nought, ttt.Nought, ttt.EmptyMark},
				{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Cross: 1, ttt.Nought: 2},
		},
		{
			name: "縦一列で勝ち",
			board: ttt.Board{
				{ttt.Nought, ttt.Cross, ttt.Cross},
				{ttt.Nought, ttt.Cross, ttt.EmptyMark},
				{ttt.Nought, ttt.EmptyMark, ttt.EmptyMark},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Nought: 1, ttt.Cross: 2},
		},
		{
			name: "斜めで勝ち",
			board: ttt.Board{
				{ttt.Cross, ttt.Nought, ttt.EmptyMark},
				{ttt.Nought, ttt.Cross, ttt.EmptyMark},
				{ttt.EmptyMark, ttt.EmptyMark, ttt.Cross},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Cross: 1, ttt.Nought: 2},
		},
		{
			name: "引き分け",
			board: ttt.Board{
				{ttt.Cross, ttt.Nought, ttt.Cross},
				{ttt.Nought, ttt.Nought, ttt.Cross},
				{ttt.Cross, ttt.Cross, ttt.Nought},
			},
			want: game.RankByAgent[ttt.Mark]{ttt.Cross: 1, ttt.Nought: 1},
		},
		{
			name: "進行中",
			board: ttt.Board{
				{ttt.Cross, ttt.EmptyMark, ttt.EmptyMark},
				{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
				{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
			},
			want: game.RankByAgent[ttt.Mark]{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := engine.RankByAgentFunc(ttt.State{Board: tc.board})
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}

			if len(got) != len(tc.want) {
				t.Fatalf("要素数の不一致: got = %v, want = %v", got, tc.want)
			}
			for agent, rank := range tc.want {
				if got[agent] != rank {
					t.Errorf("%v のrankの不一致: got = %d, want = %d", agent, got[agent], rank)
				}
			}
		})
	}
}

func TestTransition(t *testing.T) {
	engine := ttt.NewEngine()
	state := ttt.NewInitialState()

	t.Run("正常_マークが置かれ手番が替わる", func(t *testing.T) {
		next, err := engine.Logic.TransitionFunc(state, ttt.Action{Row: 1, Col: 1})
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if next.Board[1][1] != ttt.Cross {
			t.Errorf("マークの不一致: got = %v, want = %v", next.Board[1][1], ttt.Cross)
		}
		if next.Turn != ttt.Nought {
			t.Errorf("手番の不一致: got = %v, want = %v", next.Turn, ttt.Nought)
		}
	})

	t.Run("異常_置かれているマスに置く", func(t *testing.T) {
		next, err := engine.Logic.TransitionFunc(state, ttt.Action{Row: 1, Col: 1})
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		_, err = engine.Logic.TransitionFunc(next, ttt.Action{Row: 1, Col: 1})
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})

	t.Run("異常_範囲外", func(t *testing.T) {
		_, err := engine.Logic.TransitionFunc(state, ttt.Action{Row: 3, Col: 0})
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestLegalActions(t *testing.T) {
	engine := ttt.NewEngine()

	t.Run("正常_初期局面は9手", func(t *testing.T) {
		got := engine.Logic.LegalActionsFunc(ttt.NewInitialState())
		if len(got) != 9 {
			t.Errorf("合法手の数の不一致: got = %d, want = 9", len(got))
		}
	})

	t.Run("正常_決着後は合法手なし", func(t *testing.T) {
		state := ttt.State{
			Board: ttt.Board{
				{ttt.Cross, ttt.Cross, ttt.Cross},
				{ttt.Nought, ttt.Nought, ttt.EmptyMark},
				{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
			},
		}
		got := engine.Logic.LegalActionsFunc(state)
		if len(got) != 0 {
			t.Errorf("合法手の数の不一致: got = %d, want = 0", len(got))
		}
	})
}
