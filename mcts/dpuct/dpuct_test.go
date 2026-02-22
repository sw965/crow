package dpuct_test

import (
	"fmt"
	"math"
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

func TestDPUCT(t *testing.T) {
	// エージェントは int 型 (1 と 2) を使用する
	agent1 := 1
	agent2 := 2

	legalMovesByAgentFunc := func(rps RockPaperScissors) simultaneous.LegalMovesByAgent[Hand, int] {
		// ゲーム終了時は合法手なし
		if rps.Finished {
			return simultaneous.LegalMovesByAgent[Hand, int]{}
		}
		return simultaneous.LegalMovesByAgent[Hand, int]{
			agent1: HANDS,
			agent2: HANDS,
		}
	}

	moveFunc := func(rps RockPaperScissors, moves map[int]Hand) (RockPaperScissors, error) {
		return RockPaperScissors{
			Finished: true,
			Hand1:    moves[agent1],
			Hand2:    moves[agent2],
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

	gameLogic := simultaneous.Logic[RockPaperScissors, Hand, int]{
		LegalMovesByAgentFunc: legalMovesByAgentFunc,
		MoveFunc:              moveFunc,
		EqualFunc:             equalFunc,
	}

	gameEngine := simultaneous.Engine[RockPaperScissors, Hand, int]{
		Logic:           gameLogic,
		RankByAgentFunc: rankByAgentFunc,
		Agents:          []int{agent1, agent2},
	}
	gameEngine.SetStandardResultScoreByAgentFunc()

	mcts := dpuct.Engine[RockPaperScissors, Hand, int]{
		Game:         gameEngine,
		PUCBFunc:     pucb.NewAlphaGoFunc(float32(math.Sqrt(1000.0))),
		NextNodesCap: 3,
		VirtualValue: 0.5, // 並列探索時のVirtual Lossの初期値（引き分け想定の0.5など）
	}

	// PolicyとPlayoutの設定
	mcts.SetUniformPolicyFunc()
	ac := simultaneous.NewRandomActorCritic[RockPaperScissors, Hand, int]()
	
	// Playout用のRNGを渡してセットアップ
	playoutRng := randx.NewPCG()
	mcts.SetPlayout(ac, playoutRng)

	// ワーカー用のRNG配列を作成（今回は1スレッドで実行）
	rngs := randx.NewPCGs(12)

	// 探索の開始
	rootState := RockPaperScissors{}
	rootNode, err := mcts.NewNode(rootState)
	if err != nil {
		t.Fatalf("NewNode error: %v", err)
	}

	// 1回目の探索
	_, err = mcts.Search(rootNode, 19600, rngs)
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}

	fmt.Println("スレッド数:", len(rngs))
	fmt.Println("--- 19600回 シミュレーション後 ---")
	vSelectors := rootNode.VirtualSelectors()
	for agent, selector := range vSelectors {
		for a, calc := range selector {
			fmt.Printf("Agent = %d, Action = %s, Q = %.4f, Visits = %d\n", agent, a, calc.Q(), calc.Visits())
		}
	}

	fmt.Println("")

	// 2回目の探索
	_, err = mcts.Search(rootNode, 1960000, rngs)
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}

	fmt.Println("--- +196000回 シミュレーション後 ---")
	vSelectors = rootNode.VirtualSelectors()
	for agent, selector := range vSelectors {
		for a, calc := range selector {
			fmt.Printf("Agent = %d, Action = %s, Q = %.4f, Visits = %d\n", agent, a, calc.Q(), calc.Visits())
		}
	}
}