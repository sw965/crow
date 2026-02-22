package sequential_test

import (
	"github.com/sw965/crow/game"
	"github.com/sw965/crow/game/sequential"
	ttt "github.com/sw965/crow/game/sequential/tictactoe"
	"maps"
	"math"
	"strings"
	"testing"
)

func TestNewRankByAgent(t *testing.T) {
	tests := []struct {
		name           string
		agentsPerRank  [][]string
		want           game.RankByAgent[string]
		wantErr        bool
		wantErrMsgSubs []string
	}{
		//正常
		{
			name: "正常_同順なし",
			agentsPerRank: [][]string{
				{"チームA"},
				{"チームB"},
				{"チームC"},
			},
			want: game.RankByAgent[string]{
				"チームA": 1,
				"チームB": 2,
				"チームC": 3,
			},
		},
		{
			name: "正常_同順あり",
			agentsPerRank: [][]string{
				{"チームB"},
				{"チームA", "チームC"},
				{"チームD"},
				{"チームE", "チームF"},
				{"チームG"},
			},
			want: game.RankByAgent[string]{
				"チームB": 1,
				"チームA": 2,
				"チームC": 2,
				"チームD": 4,
				"チームE": 5,
				"チームF": 5,
				"チームG": 7,
			},
		},
		{
			name: "正常_一人用ゲーム",
			agentsPerRank: [][]string{
				{"プレイヤー"},
			},
			want: game.RankByAgent[string]{
				"プレイヤー": 1,
			},
		},
		//異常系
		{
			name: "異常_エージェント重複",
			// チームCが重複
			agentsPerRank: [][]string{
				[]string{"チームC"},
				[]string{"チームA", "チームB"},
				[]string{"チームD", "チームC"},
			},
			wantErr: true,
			wantErrMsgSubs: []string{
				"duplicate",
			},
		},
		{
			name: "異常_空の順位",
			agentsPerRank: [][]string{
				// 1位
				[]string{"チームA"},
				// 2位
				[]string{"チームB", "チームC"},
				// 4位
				[]string{},
				[]string{"チームF"},
			},
			wantErr: true,
			wantErrMsgSubs: []string{
				"empty",
			},
		},
		//準正常系
		{
			name:          "準正常_nil入力",
			agentsPerRank: nil,
			want:          game.RankByAgent[string]{},
		},
		{
			name:          "準正常_空スライス入力",
			agentsPerRank: [][]string{},
			want:          game.RankByAgent[string]{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Helper()
			got, err := game.NewRankByAgent(tc.agentsPerRank)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("エラーを期待したが、nilが返された")
				}

				if len(tc.wantErrMsgSubs) == 0 {
					t.Fatalf("len(wantErrMsgSubs) == 0: tc.wantErr = %t", tc.wantErr)
				}

				errMsg := err.Error()
				for _, sub := range tc.wantErrMsgSubs {
					if !strings.Contains(errMsg, sub) {
						t.Errorf("errMsg = %s, sub = %s", errMsg, sub)
					}
				}
				return
			}

			if err != nil {
				t.Fatalf("予期せぬエラーが発生した: %v", err)
			}

			if !maps.Equal(got, tc.want) {
				t.Errorf("want: %v, got: %v", tc.want, got)
			}
		})
	}
}

func TestRankByAgentValidate(t *testing.T) {
	tests := []struct {
		name           string
		rankByAgent    game.RankByAgent[string]
		wantErrMsgSubs []string
	}{
		{
			name: "正常_一人用ゲーム",
			rankByAgent: game.RankByAgent[string]{
				"プレイヤー1": 1,
			},
		},
		{
			name: "正常_同順なし",
			rankByAgent: game.RankByAgent[string]{
				"チームA": 1,
				"チームB": 2,
				"チームC": 3,
				"チームD": 4,
			},
		},
		{
			name: "正常_同順あり",
			rankByAgent: game.RankByAgent[string]{
				"チームA": 1,
				"チームB": 2,
				"チームC": 2,
				"チームD": 4,
			},
		},
		{
			name: "異常_0以下の順位_一人用ゲーム",
			rankByAgent: game.RankByAgent[string]{
				"プレイヤー": 0,
			},
			wantErrMsgSubs: []string{
				"0",
				"must be",
				">= 1",
			},
		},
		{
			name: "異常_0以下の順位_複数人ゲーム",
			rankByAgent: game.RankByAgent[string]{
				"チームA": 0,
				"チームB": 1,
				"チームC": 2,
				"チームD": 3,
			},
			wantErrMsgSubs: []string{
				"0",
				"must be",
				">= 1",
			},
		},
		{
			name: "異常_最小順位が1ではない_一人用ゲーム",
			rankByAgent: game.RankByAgent[string]{
				"プレイヤー": 2,
			},
			wantErrMsgSubs: []string{
				"must start",
				"at 1",
				"got 2",
			},
		},
		{
			name: "異常_最小順位が1ではない_複数人ゲーム",
			rankByAgent: game.RankByAgent[string]{
				"チームA": 2,
				"チームB": 3,
				"チームC": 4,
			},
			wantErrMsgSubs: []string{
				"must start",
				"at 1",
				"got 2",
			},
		},
		{
			name:        "準正常_nil入力",
			rankByAgent: nil,
		},
		{
			name:        "準正常_空値入力",
			rankByAgent: game.RankByAgent[string]{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Helper()
			err := tc.rankByAgent.Validate()
			if len(tc.wantErrMsgSubs) == 0 {
				if err != nil {
					t.Errorf("予期せぬエラーが発生: %v", err)
				}
				return
			}

			errMsg := err.Error()
			for _, sub := range tc.wantErrMsgSubs {
				if !strings.Contains(errMsg, sub) {
					t.Errorf("errMsg = %s, sub = %s", errMsg, sub)
				}
			}
		})
	}
}

func TestEngineIsEnd(t *testing.T) {
	engine := ttt.NewEngine()
	tests := []struct {
		name  string
		state ttt.State
		want  bool
	}{
		{
			name: "進行中",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Cross, ttt.Nought, ttt.Nought},
					{ttt.Nought, ttt.Cross, ttt.Nought},
					{ttt.Cross, ttt.EmptyMark, ttt.EmptyMark},
				},
				Turn: ttt.Cross,
			},
			want: false,
		},
		{
			name: "終了",
			state: ttt.State{
				Board: ttt.Board{
					{ttt.Nought, ttt.EmptyMark, ttt.EmptyMark},
					{ttt.Cross, ttt.Nought, ttt.Cross},
					{ttt.Nought, ttt.EmptyMark, ttt.Nought},
				},
			},
			want: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Helper()
			got, err := engine.IsEnd(tc.state)
			if err != nil {
				t.Fatalf("予期せぬエラー")
			}

			if tc.want != got {
				t.Errorf("want: %t, got: %t", tc.want, got)
			}
		})
	}
}

func TestEngineSetStandardResultScoreByAgentFunc(t *testing.T) {
	engine := sequential.Engine[int, int, string]{}
	engine.SetStandardResultScoreByAgentFunc()

	// 現時点で、このテストケースでは異常系は扱わない。
	// RankByAgent.Validate()の結果のみがエラーに依存している為、TestRankByAgentValidateで担保する
	tests := []struct {
		name        string
		rankByAgent game.RankByAgent[string]
		want        map[string]float32
	}{
		{
			name: "一人用ゲーム",
			rankByAgent: game.RankByAgent[string]{
				"プレイヤー": 1,
			},
			want: map[string]float32{
				"プレイヤー": 1.0,
			},
		},
		{
			name: "二人用ゲーム_同順なし",
			rankByAgent: game.RankByAgent[string]{
				"黒": 1,
				"白": 2,
			},
			want: map[string]float32{
				"黒": 1.0,
				"白": 0.0,
			},
		},
		{
			name: "二人用ゲーム_引き分け",
			rankByAgent: game.RankByAgent[string]{
				"黒": 1,
				"白": 1,
			},
			want: map[string]float32{
				"黒": 0.5,
				"白": 0.5,
			},
		},
		{
			name: "三人用ゲーム_同順なし",
			rankByAgent: game.RankByAgent[string]{
				"A": 1,
				"C": 2,
				"B": 3,
			},
			want: map[string]float32{
				"A": 1.0,
				"C": 0.5,
				"B": 0.0,
			},
		},
		{
			name: "三人用ゲーム_同順あり_全員1位",
			rankByAgent: game.RankByAgent[string]{
				"A": 1,
				"B": 1,
				"C": 1,
			},
			want: map[string]float32{
				"A": 0.5,
				"B": 0.5,
				"C": 0.5,
			},
		},
		{
			name: "三人用ゲーム_同順あり_1位1人_2位2人",
			rankByAgent: game.RankByAgent[string]{
				"A": 1,
				"B": 2,
				"C": 2,
			},
			want: map[string]float32{
				"A": 1.0,
				"B": 0.25,
				"C": 0.25,
			},
		},
		{
			name: "三人用ゲーム_同順あり_1位2人_3位1人",
			rankByAgent: game.RankByAgent[string]{
				"A": 1,
				"B": 1,
				"C": 3,
			},
			want: map[string]float32{
				"A": 0.75,
				"B": 0.75,
				"C": 0.0,
			},
		},
	}

	eps := 0.0001
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Helper()
			got, err := engine.ResultScoreByAgentFunc(tc.rankByAgent)
			if err != nil {
				t.Fatalf("予期せぬエラーが発生した: %v", err)
			}

			for k, gv := range got {
				wv := tc.want[k]
				diff := float64(math.Abs(float64(gv - wv)))
				if diff > eps {
					t.Errorf("want: %f(±%f), got: %f, key: %s", wv, eps, gv, k)
				}
			}
		})
	}
}

func TestUniformPolicyFunc(t *testing.T) {
	got, err := sequential.UniformPolicyFunc[int, string](0, []string{"戦う", "呪文", "アイテム", "逃げる"})
	if err != nil {
		t.Errorf("テスト失敗")
	}

	want := game.Policy[string]{
		"戦う":   0.25,
		"呪文":   0.25,
		"アイテム": 0.25,
		"逃げる":  0.25,
	}

	if !maps.Equal(got, want) {
		t.Errorf("want: %v, got: %v", want, got)
	}
}
