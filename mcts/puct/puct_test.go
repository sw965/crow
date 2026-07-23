package puct_test

import (
	"math"
	"testing"

	"github.com/sw965/crow/game"
	"github.com/sw965/crow/game/sequential"
	"github.com/sw965/crow/internal/ttt"
	"github.com/sw965/crow/mcts/puct"
	"github.com/sw965/crow/pucb"
	"github.com/sw965/omw/mathx/randx"
	"math/rand/v2"
)

func newTTTMCTS() puct.Engine[ttt.State, ttt.Action, ttt.Mark] {
	mcts := puct.Engine[ttt.State, ttt.Action, ttt.Mark]{
		Game:         ttt.NewEngine(),
		PUCBFunc:     pucb.NewAlphaGoFunc(1.25),
		NextNodesCap: 9,
		VirtualValue: 0.5,
	}
	mcts.SetUniformPolicyFunc()
	accr := sequential.NewRandomActorCritic[ttt.State, ttt.Action, ttt.Mark]()
	mcts.SetPlayout(accr)
	return mcts
}

// 三目並べで「次の一手で勝てる局面」を与えた場合、
// 探索が正しく機能していれば、勝つ手が最も多く訪問されるはず。
func TestSearchFindsWinningMove(t *testing.T) {
	mcts := newTTTMCTS()

	// Crossは(0,2)に置けば勝ち
	state := ttt.State{
		Board: ttt.Board{
			{ttt.Cross, ttt.Cross, ttt.EmptyMark},
			{ttt.Nought, ttt.Nought, ttt.EmptyMark},
			{ttt.EmptyMark, ttt.EmptyMark, ttt.EmptyMark},
		},
		Turn: ttt.Cross,
	}

	rootNode, err := mcts.NewNode(state)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	rngs := randx.NewPCGs(4)
	_, err = mcts.Search(rootNode, 5000, rngs)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	selector := rootNode.VirtualSelector()

	var bestAction ttt.Action
	bestVisits := -1
	for action, calc := range selector {
		if calc.Visits() > bestVisits {
			bestVisits = calc.Visits()
			bestAction = action
		}

		// 探索終了後、pending(未観測カウント)は全て解放されているはず
		if calc.Pending() != 0 {
			t.Errorf("action %v のpendingが解放されていない: got = %d, want = 0", action, calc.Pending())
		}
	}

	want := ttt.Action{Row: 0, Col: 2}
	if bestAction != want {
		t.Errorf("最多訪問の行動の不一致: got = %v, want = %v", bestAction, want)
	}

	// 勝つ手のQ値は1.0(勝ち)に近いはず
	q := float64(selector[want].Q())
	if math.Abs(q-1.0) > 0.05 {
		t.Errorf("勝つ手のQ値の不一致: got = %.4f, want = 1.0(±0.05)", q)
	}
}

func TestNewPolicyValueFunc(t *testing.T) {
	mcts := newTTTMCTS()
	rngs := randx.NewPCGs(2)
	pvFunc := mcts.NewPolicyValueFunc(1000, rngs)

	state := ttt.NewInitialState()
	legalActions := mcts.Game.Logic.LegalActionsFunc(state)

	policy, value, err := pvFunc(state, legalActions)
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	if len(policy) != 9 {
		t.Fatalf("policyの要素数の不一致: got = %d, want = 9", len(policy))
	}

	// policyは確率分布(合計1)
	var sum float32
	for _, p := range policy {
		sum += p
	}
	if math.Abs(float64(sum)-1.0) > 0.0001 {
		t.Errorf("policyの合計の不一致: got = %f, want = 1.0", sum)
	}

	// 三目並べの初期局面の価値は、引き分け(0.5)前後になるはず
	if value < 0.0 || value > 1.0 {
		t.Errorf("価値が範囲外: got = %f, want = [0.0, 1.0]", value)
	}
}

func TestEngineValidate(t *testing.T) {
	t.Run("正常", func(t *testing.T) {
		mcts := newTTTMCTS()
		if err := mcts.Validate(); err != nil {
			t.Errorf("予期せぬエラー: %v", err)
		}
	})

	t.Run("異常_PUCBFuncがnil", func(t *testing.T) {
		mcts := newTTTMCTS()
		mcts.PUCBFunc = nil
		if err := mcts.Validate(); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})

	t.Run("異常_NextNodesCapが0", func(t *testing.T) {
		mcts := newTTTMCTS()
		mcts.NextNodesCap = 0
		if err := mcts.Validate(); err == nil {
			t.Fatal("エラーを期待したが、nilが返された")
		}
	})
}

// 終了しないゲームでも、MaxDepthを設定すれば探索が打ち切られる事を確認する。
type endlessState struct {
	Pos int
}

func newEndlessMCTS(maxDepth int) puct.Engine[endlessState, int, string] {
	// 2つの状態を行き来し続ける、終了しないゲーム
	gameEngine := sequential.Engine[endlessState, int, string]{
		Logic: sequential.Logic[endlessState, int, string]{
			LegalActionsFunc: func(endlessState) []int { return []int{0, 1} },
			TransitionFunc: func(s endlessState, a int) (endlessState, error) {
				return endlessState{Pos: (s.Pos + 1) % 2}, nil
			},
			EqualFunc:        func(a, b endlessState) bool { return a == b },
			CurrentAgentFunc: func(endlessState) string { return "A" },
		},
		RankByAgentFunc: func(endlessState) (game.RankByAgent[string], error) {
			return game.RankByAgent[string]{}, nil
		},
		Agents: []string{"A"},
	}
	gameEngine.SetStandardResultScoreByAgentFunc()

	mcts := puct.Engine[endlessState, int, string]{
		Game:         gameEngine,
		PUCBFunc:     pucb.NewAlphaGoFunc(1.25),
		NextNodesCap: 2,
		VirtualValue: 0.5,
		MaxDepth:     maxDepth,
		// 終了しないゲームなので、プレイアウトではなく固定値でリーフを評価する
		LeafNodeEvalByAgentFunc: func(s endlessState, rng *rand.Rand) (puct.LeafNodeEvalByAgent[string], error) {
			return puct.LeafNodeEvalByAgent[string]{"A": 0.5}, nil
		},
	}
	mcts.SetUniformPolicyFunc()
	return mcts
}

func TestSearchMaxDepth(t *testing.T) {
	const maxDepth = 5
	mcts := newEndlessMCTS(maxDepth)

	rootNode, err := mcts.NewNode(endlessState{})
	if err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}

	rng := randx.NewPCG()

	// シミュレーションを重ねても、1回の探索の深さはMaxDepthを超えない
	for i := 0; i < 200; i++ {
		evals, depth, err := mcts.SelectExpansionBackward(rootNode, 0, rng)
		if err != nil {
			t.Fatalf("予期せぬエラー: %v", err)
		}
		if depth > maxDepth {
			t.Fatalf("深さがMaxDepthを超えた: depth = %d, MaxDepth = %d", depth, maxDepth)
		}
		if math.Abs(float64(evals["A"])-0.5) > 0.0001 {
			t.Fatalf("評価値の不一致: got = %f, want = 0.5", evals["A"])
		}
	}

	// Searchも問題なく完了する
	if _, err := mcts.Search(rootNode, 1000, randx.NewPCGs(4)); err != nil {
		t.Fatalf("予期せぬエラー: %v", err)
	}
}
