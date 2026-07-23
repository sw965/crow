package dpuct_test

import (
	"math"
	"strings"
	"testing"

	"github.com/sw965/crow/game"
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/mcts/dpuct"
	"github.com/sw965/crow/pucb"
	"github.com/sw965/omw/mathx/randx"
)

type Hand string

const (
	ROCK     Hand = "グー"
	PAPER    Hand = "パー"
	SCISSORS Hand = "チョキ"
)

var HANDS = []Hand{ROCK, PAPER, SCISSORS}

// 状態 S
type RockPaperScissors struct {
	Finished bool
	Hand1    Hand
	Hand2    Hand
}

func newRPSEngine(agent1, agent2 int) simultaneous.Engine[RockPaperScissors, Hand, int] {
	legalActionsByAgentFunc := func(rps RockPaperScissors) simultaneous.LegalActionsByAgent[Hand, int] {
		// ゲーム終了時は合法手なし
		if rps.Finished {
			return simultaneous.LegalActionsByAgent[Hand, int]{}
		}
		return simultaneous.LegalActionsByAgent[Hand, int]{
			agent1: HANDS,
			agent2: HANDS,
		}
	}

	transitionFunc := func(rps RockPaperScissors, actions simultaneous.JointAction[Hand, int]) (RockPaperScissors, error) {
		return RockPaperScissors{
			Finished: true,
			Hand1:    actions[agent1],
			Hand2:    actions[agent2],
		}, nil
	}

	equalFunc := func(rps1, rps2 RockPaperScissors) bool {
		return rps1 == rps2
	}

	rankByAgentFunc := func(rps RockPaperScissors) (game.RankByAgent[int], error) {
		if !rps.Finished {
			return game.RankByAgent[int]{}, nil
		}

		if rps.Hand1 == rps.Hand2 {
			// 引き分けの場合は同順位
			return game.RankByAgent[int]{agent1: 1, agent2: 1}, nil
		}

		h1 := rps.Hand1
		h2 := rps.Hand2

		// プレイヤー1が勝つ条件
		if (h1 == ROCK && h2 == SCISSORS) ||
			(h1 == SCISSORS && h2 == PAPER) ||
			(h1 == PAPER && h2 == ROCK) {
			return game.RankByAgent[int]{agent1: 1, agent2: 2}, nil
		}

		// それ以外はプレイヤー2の勝ち
		return game.RankByAgent[int]{agent1: 2, agent2: 1}, nil
	}

	engine := simultaneous.Engine[RockPaperScissors, Hand, int]{
		Logic: simultaneous.Logic[RockPaperScissors, Hand, int]{
			LegalActionsByAgentFunc: legalActionsByAgentFunc,
			TransitionFunc:          transitionFunc,
			EqualFunc:               equalFunc,
		},
		RankByAgentFunc: rankByAgentFunc,
		Agents:          []int{agent1, agent2},
	}
	engine.SetStandardResultScoreByAgentFunc()
	return engine
}

// じゃんけんのナッシュ均衡は各手を1/3で出す事。
// 探索が正しく機能していれば、十分なシミュレーション後に、
// 各手の訪問比率は1/3に、Q値は0.5(引き分け相当)に近づくはず。
func TestDPUCT(t *testing.T) {
	agent1 := 1
	agent2 := 2
	gameEngine := newRPSEngine(agent1, agent2)

	mcts := dpuct.Engine[RockPaperScissors, Hand, int]{
		Game:         gameEngine,
		PUCBFunc:     pucb.NewAlphaGoFunc(float32(math.Sqrt(1000.0))),
		NextNodesCap: 3,
		VirtualValue: 0.5, // 並列探索時のVirtual Lossの初期値(引き分け想定の0.5)
	}

	mcts.SetUniformPolicyFunc()
	accr := simultaneous.NewRandomActorCritic[RockPaperScissors, Hand, int]()
	mcts.SetPlayout(accr)

	// マルチスレッドで探索する
	rngs := randx.NewPCGs(4)

	rootState := RockPaperScissors{}
	rootNode, err := mcts.NewNode(rootState)
	if err != nil {
		t.Fatalf("NewNode error: %v", err)
	}

	simulations := 30000
	evals, err := mcts.Search(rootNode, simulations, rngs)
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}

	// ルート評価値は引き分け相当の0.5付近
	const evalEps = 0.05
	for agent, eval := range evals {
		if math.Abs(float64(eval)-0.5) > evalEps {
			t.Errorf("Agent %d のルート評価値の不一致: got = %.4f, want = 0.5(±%.2f)", agent, eval, evalEps)
		}
	}

	const ratioEps = 0.05
	const qEps = 0.05
	vSelectors := rootNode.VirtualSelectors()

	if len(vSelectors) != 2 {
		t.Fatalf("エージェント数の不一致: got = %d, want = 2", len(vSelectors))
	}

	for agent, selector := range vSelectors {
		if len(selector) != len(HANDS) {
			t.Fatalf("Agent %d の行動数の不一致: got = %d, want = %d", agent, len(selector), len(HANDS))
		}

		sumVisits := selector.SumVisits()
		if sumVisits < simulations {
			t.Errorf("Agent %d の合計訪問数が不足: got = %d, want >= %d", agent, sumVisits, simulations)
		}

		for hand, calc := range selector {
			// 各手の訪問比率は1/3に近い
			ratio := float64(calc.Visits()) / float64(sumVisits)
			if math.Abs(ratio-1.0/3.0) > ratioEps {
				t.Errorf("Agent %d, %s の訪問比率の不一致: got = %.4f, want = 0.333(±%.2f)", agent, hand, ratio, ratioEps)
			}

			// 各手のQ値は0.5に近い
			q := float64(calc.Q())
			if math.Abs(q-0.5) > qEps {
				t.Errorf("Agent %d, %s のQ値の不一致: got = %.4f, want = 0.5(±%.2f)", agent, hand, q, qEps)
			}

			// 探索終了後、pending(未観測カウント)は全て解放されているはず
			if calc.Pending() != 0 {
				t.Errorf("Agent %d, %s のpendingが解放されていない: got = %d, want = 0", agent, hand, calc.Pending())
			}
		}
	}

	// 追加探索しても、正しく訪問数が積み上がる
	_, err = mcts.Search(rootNode, simulations, rngs)
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}

	vSelectors = rootNode.VirtualSelectors()
	for agent, selector := range vSelectors {
		sumVisits := selector.SumVisits()
		if sumVisits < 2*simulations {
			t.Errorf("Agent %d の合計訪問数が不足: got = %d, want >= %d", agent, sumVisits, 2*simulations)
		}
	}
}

func TestDPUCTNewPolicyValueFunc(t *testing.T) {
	agent1 := 1
	agent2 := 2
	gameEngine := newRPSEngine(agent1, agent2)

	mcts := dpuct.Engine[RockPaperScissors, Hand, int]{
		Game:         gameEngine,
		PUCBFunc:     pucb.NewAlphaGoFunc(float32(math.Sqrt(1000.0))),
		NextNodesCap: 3,
		VirtualValue: 0.5,
	}
	mcts.SetUniformPolicyFunc()
	accr := simultaneous.NewRandomActorCritic[RockPaperScissors, Hand, int]()
	mcts.SetPlayout(accr)

	rngs := randx.NewPCGs(2)
	pvFunc := mcts.NewPolicyValueFunc(3000, rngs)

	rootState := RockPaperScissors{}
	legalActionsByAgent := gameEngine.Logic.LegalActionsByAgentFunc(rootState)
	policyByAgent, valueByAgent, err := pvFunc(rootState, legalActionsByAgent)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	for _, agent := range []int{agent1, agent2} {
		policy := policyByAgent[agent]
		if len(policy) != len(HANDS) {
			t.Fatalf("Agent %d のpolicyの要素数の不一致: got = %d, want = %d", agent, len(policy), len(HANDS))
		}

		// policyは確率分布(合計1)
		var sum float32
		for _, p := range policy {
			sum += p
		}
		if math.Abs(float64(sum)-1.0) > 0.0001 {
			t.Errorf("Agent %d のpolicyの合計の不一致: got = %f, want = 1.0", agent, sum)
		}

		// 価値は0.5付近
		if math.Abs(float64(valueByAgent[agent])-0.5) > 0.1 {
			t.Errorf("Agent %d の価値の不一致: got = %f, want = 0.5(±0.1)", agent, valueByAgent[agent])
		}
	}
}

func TestDPUCTSelectionErrorReleasesPending(t *testing.T) {
	const (
		agent1 = 1
		agent2 = 2
	)
	gameEngine := newRPSEngine(agent1, agent2)
	mcts := dpuct.Engine[RockPaperScissors, Hand, int]{
		Game:         gameEngine,
		PUCBFunc:     pucb.NewAlphaGoFunc(1),
		NextNodesCap: 3,
		VirtualValue: 0.5,
	}
	mcts.SetUniformPolicyFunc()
	mcts.SetPlayout(simultaneous.NewRandomActorCritic[RockPaperScissors, Hand, int]())

	rootNode, err := mcts.NewNode(RockPaperScissors{})
	if err != nil {
		t.Fatalf("NewNode error: %v", err)
	}
	selectors := rootNode.VirtualSelectors()
	for _, calculator := range selectors[agent2] {
		calculator.Func = nil
	}

	_, _, err = mcts.SelectExpansionBackward(rootNode, 0, randx.NewPCG())
	if err == nil {
		t.Fatal("エラーを期待したが、nilが返された")
	}
	if !strings.Contains(err.Error(), "Func") {
		t.Fatalf("エラーメッセージが不十分: %v", err)
	}

	for agent, selector := range rootNode.VirtualSelectors() {
		for action, calculator := range selector {
			if got := calculator.Pending(); got != 0 {
				t.Errorf("pendingが解放されていない: agent=%d action=%s got=%d", agent, action, got)
			}
		}
	}
}
