package simultaneous_test

import (
	"math"
	"strings"
	"testing"

	"github.com/sw965/crow/game"
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/omw/mathx/randx"
)

type Hand string

const (
	ROCK     Hand = "グー"
	PAPER    Hand = "パー"
	SCISSORS Hand = "チョキ"
)

var HANDS = []Hand{ROCK, PAPER, SCISSORS}

type RockPaperScissors struct {
	Finished bool
	Hand1    Hand
	Hand2    Hand
}

const (
	agent1 = 1
	agent2 = 2
)

func newRPSEngine() simultaneous.Engine[RockPaperScissors, Hand, int] {
	engine := simultaneous.Engine[RockPaperScissors, Hand, int]{
		Logic: simultaneous.Logic[RockPaperScissors, Hand, int]{
			LegalActionsByAgentFunc: func(rps RockPaperScissors) simultaneous.LegalActionsByAgent[Hand, int] {
				if rps.Finished {
					return simultaneous.LegalActionsByAgent[Hand, int]{}
				}
				return simultaneous.LegalActionsByAgent[Hand, int]{
					agent1: HANDS,
					agent2: HANDS,
				}
			},
			TransitionFunc: func(rps RockPaperScissors, actions simultaneous.JointAction[Hand, int]) (RockPaperScissors, error) {
				return RockPaperScissors{
					Finished: true,
					Hand1:    actions[agent1],
					Hand2:    actions[agent2],
				}, nil
			},
			EqualFunc: func(a, b RockPaperScissors) bool { return a == b },
		},
		RankByAgentFunc: func(rps RockPaperScissors) (game.RankByAgent[int], error) {
			if !rps.Finished {
				return game.RankByAgent[int]{}, nil
			}
			if rps.Hand1 == rps.Hand2 {
				return game.RankByAgent[int]{agent1: 1, agent2: 1}, nil
			}
			h1, h2 := rps.Hand1, rps.Hand2
			if (h1 == ROCK && h2 == SCISSORS) || (h1 == SCISSORS && h2 == PAPER) || (h1 == PAPER && h2 == ROCK) {
				return game.RankByAgent[int]{agent1: 1, agent2: 2}, nil
			}
			return game.RankByAgent[int]{agent1: 2, agent2: 1}, nil
		},
		Agents: []int{agent1, agent2},
	}
	engine.SetStandardResultScoreByAgentFunc()
	return engine
}

func TestUniformPolicyNoValueFunc(t *testing.T) {
	t.Run("正常", func(t *testing.T) {
		legalActionsByAgent := simultaneous.LegalActionsByAgent[Hand, int]{
			agent1: HANDS,
			agent2: {ROCK},
		}

		policyByAgent, valueByAgent, err := simultaneous.UniformPolicyNoValueFunc[RockPaperScissors](RockPaperScissors{}, legalActionsByAgent)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}

		for _, p := range policyByAgent[agent1] {
			if math.Abs(float64(p)-1.0/3.0) > 0.0001 {
				t.Errorf("agent1の確率の不一致: got = %f, want = 0.333", p)
			}
		}
		if p := policyByAgent[agent2][ROCK]; math.Abs(float64(p)-1.0) > 0.0001 {
			t.Errorf("agent2の確率の不一致: got = %f, want = 1.0", p)
		}

		for agent, v := range valueByAgent {
			if v != 0.0 {
				t.Errorf("agent %d のvalueの不一致: got = %f, want = 0.0", agent, v)
			}
		}
	})

	t.Run("異常_合法手が空のエージェント", func(t *testing.T) {
		legalActionsByAgent := simultaneous.LegalActionsByAgent[Hand, int]{
			agent1: HANDS,
			agent2: {},
		}
		_, _, err := simultaneous.UniformPolicyNoValueFunc[RockPaperScissors](RockPaperScissors{}, legalActionsByAgent)
		if err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
		if !strings.Contains(err.Error(), "合法手がありません") {
			t.Errorf("エラーメッセージが不十分: %s", err.Error())
		}
	})
}

func TestEnginePlayouts(t *testing.T) {
	engine := newRPSEngine()
	accr := simultaneous.NewRandomActorCritic[RockPaperScissors, Hand, int]()
	rngs := randx.NewPCGs(2)

	n := 50
	inits := make([]RockPaperScissors, n)

	finals, err := engine.Playouts(inits, accr, rngs)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	if len(finals) != n {
		t.Fatalf("len(finals)の不一致: got = %d, want = %d", len(finals), n)
	}

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
	engine := newRPSEngine()
	accr := simultaneous.NewRandomActorCritic[RockPaperScissors, Hand, int]()
	rngs := randx.NewPCGs(2)

	n := 20
	inits := make([]RockPaperScissors, n)

	records, err := engine.RecordPlayouts(inits, accr, rngs, 1)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	if len(records) != n {
		t.Fatalf("len(records)の不一致: got = %d, want = %d", len(records), n)
	}

	for i, record := range records {
		// じゃんけんは1手で終了する
		if len(record.Steps) != 1 {
			t.Fatalf("records[%d]の手数の不一致: got = %d, want = 1", i, len(record.Steps))
		}

		// 記録された同時行動が最終状態と一致する
		step := record.Steps[0]
		if record.FinalState.Hand1 != step.JointAction[agent1] || record.FinalState.Hand2 != step.JointAction[agent2] {
			t.Errorf("records[%d]のJointActionとFinalStateが一致しない", i)
		}

		// 二人ゲームなのでスコアの合計は1
		sum := record.ResultScoreByAgent[agent1] + record.ResultScoreByAgent[agent2]
		if math.Abs(float64(sum)-1.0) > 0.0001 {
			t.Errorf("records[%d]のスコア合計の不一致: got = %f, want = 1.0", i, sum)
		}
	}
}

// 終了しないゲームでも、MaxStepsを設定すればプレイアウトがエラーで止まる事を確認する
func TestEnginePlayouts_MaxSteps(t *testing.T) {
	endlessEngine := simultaneous.Engine[int, int, string]{
		Logic: simultaneous.Logic[int, int, string]{
			LegalActionsByAgentFunc: func(int) simultaneous.LegalActionsByAgent[int, string] {
				return simultaneous.LegalActionsByAgent[int, string]{"A": {0}}
			},
			TransitionFunc: func(s int, actions simultaneous.JointAction[int, string]) (int, error) { return s, nil },
			EqualFunc:      func(a, b int) bool { return a == b },
		},
		RankByAgentFunc: func(int) (game.RankByAgent[string], error) {
			return game.RankByAgent[string]{}, nil
		},
		Agents:   []string{"A"},
		MaxSteps: 10,
	}
	endlessEngine.SetStandardResultScoreByAgentFunc()

	accr := simultaneous.NewRandomActorCritic[int, int, string]()
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
	engine := newRPSEngine()

	accr1 := simultaneous.NewRandomActorCritic[RockPaperScissors, Hand, int]()
	accr1.Name = "rand1"
	accr2 := simultaneous.NewRandomActorCritic[RockPaperScissors, Hand, int]()
	accr2.Name = "rand2"

	n := 10
	inits := make([]RockPaperScissors, n)

	recorder, err := engine.NewCrossPlayoutRecorder(inits, []simultaneous.ActorCritic[RockPaperScissors, Hand, int]{accr1, accr2}, 2)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	records, err := recorder.Collect()
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	wantGames := 2 * n
	if len(records) != wantGames {
		t.Fatalf("len(records)の不一致: got = %d, want = %d", len(records), wantGames)
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
	_, err = engine.NewCrossPlayoutRecorder(inits, []simultaneous.ActorCritic[RockPaperScissors, Hand, int]{accr1}, 2)
	if err == nil {
		t.Fatal("エラーを期待したが、nilが返された")
	}
}

func TestEngineCrossPlayoutRecorderNext(t *testing.T) {
	engine := newRPSEngine()

	accr1 := simultaneous.NewRandomActorCritic[RockPaperScissors, Hand, int]()
	accr1.Name = "rand1"
	accr2 := simultaneous.NewRandomActorCritic[RockPaperScissors, Hand, int]()
	accr2.Name = "rand2"

	n := 5
	inits := make([]RockPaperScissors, n)

	recorder, err := engine.NewCrossPlayoutRecorder(inits, []simultaneous.ActorCritic[RockPaperScissors, Hand, int]{accr1, accr2}, 2)
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

func TestRecordElmoSteps(t *testing.T) {
	record := simultaneous.Record[int, int, string]{
		Steps: []simultaneous.Step[int, int, string]{
			{
				State:        0,
				ValueByAgent: simultaneous.ValueByAgent[string]{"A": 0.2, "B": 0.8},
			},
		},
		ResultScoreByAgent: game.ResultScoreByAgent[string]{"A": 1.0, "B": 0.0},
	}

	tests := []struct {
		name  string
		alpha float32
		wantA float32
		wantB float32
	}{
		// alpha*Z + (1-alpha)*V を、エージェント毎に計算する
		{name: "正常_ブレンド", alpha: 0.5, wantA: 0.6, wantB: 0.4},
		{name: "正常_境界_結果のみ", alpha: 1.0, wantA: 1.0, wantB: 0.0},
		{name: "正常_境界_探索値のみ", alpha: 0.0, wantA: 0.2, wantB: 0.8},
		// 範囲外はクリップされる
		{name: "準正常_alphaが1超過", alpha: 2.0, wantA: 1.0, wantB: 0.0},
		{name: "準正常_alphaが負", alpha: -1.0, wantA: 0.2, wantB: 0.8},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			elmoSteps := record.ElmoSteps(tc.alpha)
			if len(elmoSteps) != 1 {
				t.Fatalf("len(elmoSteps)の不一致: got = %d, want = 1", len(elmoSteps))
			}

			gotA := elmoSteps[0].ValueByAgent["A"]
			if math.Abs(float64(gotA-tc.wantA)) > 0.0001 {
				t.Errorf("Aの価値の不一致: got = %f, want = %f", gotA, tc.wantA)
			}

			gotB := elmoSteps[0].ValueByAgent["B"]
			if math.Abs(float64(gotB-tc.wantB)) > 0.0001 {
				t.Errorf("Bの価値の不一致: got = %f, want = %f", gotB, tc.wantB)
			}
		})
	}
}
