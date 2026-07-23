package sequential_test

import (
	"maps"
	"math"
	"strings"
	"testing"

	"github.com/sw965/crow/game"
	"github.com/sw965/crow/game/sequential"
	"github.com/sw965/crow/internal/ttt"
	"github.com/sw965/omw/mathx/randx"
)

func TestEngineIsTerminal(t *testing.T) {
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
			got, err := engine.IsTerminal(tc.state)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
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

	if engine.ResultScoreByAgentFunc == nil {
		t.Fatal("ResultScoreByAgentFuncが設定されていない")
	}

	// スコア計算の詳細は、gameパッケージのTestStandardResultScoreByAgentFuncで担保する。
	// ここでは、標準のスコア関数が設定されている事だけを確認する。
	got, err := engine.ResultScoreByAgentFunc(game.RankByAgent[string]{"黒": 1, "白": 2})
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	want := game.ResultScoreByAgent[string]{"黒": 1.0, "白": 0.0}
	if !maps.Equal(got, want) {
		t.Errorf("want: %v, got: %v", want, got)
	}
}

func TestUniformPolicyFunc(t *testing.T) {
	got, err := sequential.UniformPolicyFunc[int, string](0, []string{"戦う", "呪文", "アイテム", "逃げる"})
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
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

	t.Run("異常_合法手が空", func(t *testing.T) {
		_, err := sequential.UniformPolicyFunc[int, string](0, []string{})
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

func TestEnginePlayouts(t *testing.T) {
	engine := ttt.NewEngine()
	accr := sequential.NewRandomActorCritic[ttt.State, ttt.Action, ttt.Mark]()
	rngs := randx.NewPCGs(2)

	n := 50
	inits := make([]ttt.State, n)
	for i := range inits {
		inits[i] = ttt.NewInitialState()
	}

	finals, err := engine.Playouts(inits, accr, rngs)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	if len(finals) != n {
		t.Fatalf("len(finals)の不一致: got = %d, want = %d", len(finals), n)
	}

	// 全ての最終状態は、ゲームが終了しているはず
	for i, final := range finals {
		isEnd, err := engine.IsTerminal(final)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !isEnd {
			t.Errorf("finals[%d]がゲーム終了状態ではない: %v", i, final)
		}
	}
}

func TestEngineRecordPlayouts(t *testing.T) {
	engine := ttt.NewEngine()
	accr := sequential.NewRandomActorCritic[ttt.State, ttt.Action, ttt.Mark]()
	rngs := randx.NewPCGs(2)

	n := 20
	inits := make([]ttt.State, n)
	for i := range inits {
		inits[i] = ttt.NewInitialState()
	}

	records, err := engine.RecordPlayouts(inits, accr, rngs, 9)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	if len(records) != n {
		t.Fatalf("len(records)の不一致: got = %d, want = %d", len(records), n)
	}

	for i, record := range records {
		// 最終状態は終了しているはず
		isEnd, err := engine.IsTerminal(record.FinalState)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !isEnd {
			t.Errorf("records[%d].FinalStateがゲーム終了状態ではない", i)
		}

		// 三目並べは5手以上9手以下で終了する
		if len(record.Steps) < 5 || len(record.Steps) > 9 {
			t.Errorf("records[%d]の手数が不正: got = %d, want = [5, 9]", i, len(record.Steps))
		}

		// スコアは全エージェント分あり、二人ゲームの合計は1になる
		if len(record.ResultScoreByAgent) != 2 {
			t.Fatalf("records[%d]のスコアのエージェント数が不正: got = %d, want = 2", i, len(record.ResultScoreByAgent))
		}
		sum := record.ResultScoreByAgent[ttt.Cross] + record.ResultScoreByAgent[ttt.Nought]
		if math.Abs(float64(sum-1.0)) > 0.0001 {
			t.Errorf("records[%d]のスコア合計の不一致: got = %f, want = 1.0", i, sum)
		}

		// 記録された遷移を辿ると、最終状態に到達するはず
		state := inits[i]
		for stepIdx, step := range record.Steps {
			if !engine.Logic.EqualFunc(state, step.State) {
				t.Fatalf("records[%d].Steps[%d].Stateが遷移と一致しない", i, stepIdx)
			}
			state, err = engine.Logic.TransitionFunc(state, step.Action)
			if err != nil {
				t.Fatalf("予期せぬエラー: %v", err)
			}
		}
		if !engine.Logic.EqualFunc(state, record.FinalState) {
			t.Errorf("records[%d]: 遷移を辿った結果がFinalStateと一致しない", i)
		}
	}
}

func TestEngineRecordElmoSteps(t *testing.T) {
	record := sequential.Record[int, int, string]{
		Steps: []sequential.Step[int, int, string]{
			{State: 0, Agent: "A", Action: 1, Value: 0.2},
		},
		ResultScoreByAgent: game.ResultScoreByAgent[string]{"A": 1.0},
	}

	tests := []struct {
		name  string
		alpha float32
		want  float32
	}{
		// alpha*Z + (1-alpha)*V
		{name: "正常_ブレンド", alpha: 0.5, want: 0.6},
		{name: "正常_境界_結果のみ", alpha: 1.0, want: 1.0},
		{name: "正常_境界_探索値のみ", alpha: 0.0, want: 0.2},
		// 範囲外はクリップされる
		{name: "準正常_alphaが1超過", alpha: 2.0, want: 1.0},
		{name: "準正常_alphaが負", alpha: -1.0, want: 0.2},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			elmoSteps := record.ElmoSteps(tc.alpha)
			if len(elmoSteps) != 1 {
				t.Fatalf("len(elmoSteps)の不一致: got = %d, want = 1", len(elmoSteps))
			}
			got := elmoSteps[0].Value
			if math.Abs(float64(got-tc.want)) > 0.0001 {
				t.Errorf("Valueの不一致: got = %f, want = %f", got, tc.want)
			}
		})
	}
}

// 終了しないゲームでも、MaxStepsを設定すればプレイアウトがエラーで止まる事を確認する
func TestEnginePlayouts_MaxSteps(t *testing.T) {
	// 常に同じ状態に戻る、終了しないゲーム
	endlessEngine := sequential.Engine[int, int, string]{
		Logic: sequential.Logic[int, int, string]{
			LegalActionsFunc: func(int) []int { return []int{0} },
			TransitionFunc:   func(s int, a int) (int, error) { return s, nil },
			EqualFunc:        func(a, b int) bool { return a == b },
			CurrentAgentFunc: func(int) string { return "A" },
		},
		RankByAgentFunc: func(int) (game.RankByAgent[string], error) {
			return game.RankByAgent[string]{}, nil
		},
		Agents:   []string{"A"},
		MaxSteps: 10,
	}
	endlessEngine.SetStandardResultScoreByAgentFunc()

	accr := sequential.NewRandomActorCritic[int, int, string]()
	rngs := randx.NewPCGs(1)

	_, err := endlessEngine.Playouts([]int{0}, accr, rngs)
	if err == nil {
		t.Fatal("エラーを期待したが、nilが返された")
	}
	if !strings.Contains(err.Error(), "MaxSteps") {
		t.Errorf("エラーメッセージが不十分: %s", err.Error())
	}

	_, err = endlessEngine.RecordPlayouts([]int{0}, accr, rngs, 16)
	if err == nil {
		t.Fatal("エラーを期待したが、nilが返された")
	}
	if !strings.Contains(err.Error(), "MaxSteps") {
		t.Errorf("エラーメッセージが不十分: %s", err.Error())
	}
}

func TestEngineCrossPlayoutRecorder(t *testing.T) {
	engine := ttt.NewEngine()

	accr1 := sequential.NewRandomActorCritic[ttt.State, ttt.Action, ttt.Mark]()
	accr1.Name = "rand1"
	accr2 := sequential.NewRandomActorCritic[ttt.State, ttt.Action, ttt.Mark]()
	accr2.Name = "rand2"

	n := 10
	inits := make([]ttt.State, n)
	for i := range inits {
		inits[i] = ttt.NewInitialState()
	}

	recorder, err := engine.NewCrossPlayoutRecorder(inits, []sequential.ActorCritic[ttt.State, ttt.Action, ttt.Mark]{accr1, accr2}, 2)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	records, err := recorder.Collect()
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	// 2つのActorCriticの順列は2通り。それぞれn試合行われる
	wantGames := 2 * n
	if len(records) != wantGames {
		t.Fatalf("len(records)の不一致: got = %d, want = %d", len(records), wantGames)
	}
	if recorder.NumGames() != wantGames {
		t.Errorf("NumGames()の不一致: got = %d, want = %d", recorder.NumGames(), wantGames)
	}

	// 各ActorCriticは全試合に参加している
	numGamesByName := recorder.NumGamesByActorCriticName()
	for _, name := range []game.ActorCriticName{"rand1", "rand2"} {
		if numGamesByName[name] != wantGames {
			t.Errorf("%s の試合数の不一致: got = %d, want = %d", name, numGamesByName[name], wantGames)
		}
	}

	// 二人ゲームなので、全ActorCriticの合計スコアは試合数と一致する
	totalByName := recorder.TotalScoreByActorCriticName()
	var total float32
	for _, v := range totalByName {
		total += v
	}
	if math.Abs(float64(total)-float64(wantGames)) > 0.0001 {
		t.Errorf("合計スコアの不一致: got = %f, want = %d", total, wantGames)
	}

	// 異常系: ActorCriticが不足
	_, err = engine.NewCrossPlayoutRecorder(inits, []sequential.ActorCritic[ttt.State, ttt.Action, ttt.Mark]{accr1}, 2)
	if err == nil {
		t.Fatal("エラーを期待したが、nilが返された")
	}
}

func TestEngineCrossPlayoutRecorderNext(t *testing.T) {
	engine := ttt.NewEngine()

	accr1 := sequential.NewRandomActorCritic[ttt.State, ttt.Action, ttt.Mark]()
	accr1.Name = "rand1"
	accr2 := sequential.NewRandomActorCritic[ttt.State, ttt.Action, ttt.Mark]()
	accr2.Name = "rand2"

	n := 5
	inits := make([]ttt.State, n)
	for i := range inits {
		inits[i] = ttt.NewInitialState()
	}

	recorder, err := engine.NewCrossPlayoutRecorder(inits, []sequential.ActorCritic[ttt.State, ttt.Action, ttt.Mark]{accr1, accr2}, 2)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	// まだ1試合も行われていない場合、平均スコアはエラー
	if _, err := recorder.AverageScoreByActorCriticName(); err == nil {
		t.Fatal("エラーを期待したが、nilが返された")
	}

	// 2つのActorCriticの並びは2通り。Nextは1回の呼び出しで1並び分(n試合)を返す
	numPerms := 2
	for i := 1; i <= numPerms; i++ {
		records, hasNext, err := recorder.Next()
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if !hasNext {
			t.Fatalf("%d回目のNextで、次があるはずなのに「次なし」が返された", i)
		}
		if len(records) != n {
			t.Fatalf("%d回目のNextの記録数の不一致: got = %d, want = %d", i, len(records), n)
		}
		if recorder.NumGames() != i*n {
			t.Fatalf("%d回目のNext後の試合数の不一致: got = %d, want = %d", i, recorder.NumGames(), i*n)
		}
	}

	// 全ての並びを消費した後は「次なし」
	records, hasNext, err := recorder.Next()
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
	if hasNext || records != nil {
		t.Errorf("消費後のNextの不一致: records = %v, hasNext = %t, want = nil, false", records, hasNext)
	}

	// 平均スコア = 合計スコア / 参加試合数
	totalByName := recorder.TotalScoreByActorCriticName()
	numGamesByName := recorder.NumGamesByActorCriticName()
	avgByName, err := recorder.AverageScoreByActorCriticName()
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	for _, name := range []game.ActorCriticName{"rand1", "rand2"} {
		want := totalByName[name] / float32(numGamesByName[name])
		if math.Abs(float64(avgByName[name]-want)) > 0.0001 {
			t.Errorf("%s の平均スコアの不一致: got = %f, want = %f", name, avgByName[name], want)
		}
	}

	// 二人ゲームなので、全ActorCriticの平均スコアの合計は1になる
	sum := avgByName["rand1"] + avgByName["rand2"]
	if math.Abs(float64(sum)-1.0) > 0.0001 {
		t.Errorf("平均スコアの合計の不一致: got = %f, want = 1.0", sum)
	}
}
