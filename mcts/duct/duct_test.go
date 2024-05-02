package duct_test

import (
	"fmt"
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/mcts/duct"
	"github.com/sw965/crow/ucb"
	"github.com/sw965/omw"
	"math"
	"testing"
)

type Hand string

const (
	ROCK     = "グー"
	PAPER    = "パー"
	SCISSORS = "チョキ"
)

type Hands []Hand

var HANDS = Hands{ROCK, PAPER, SCISSORS}

type Handss []Hands

type RockPaperScissors struct {
	Hand1 Hand
	Hand2 Hand
}

func TestDUCT(t *testing.T) {
	r := omw.NewMt19937()

	legalActionss := func(rps *RockPaperScissors) Handss {
		return Handss{HANDS, Hands{ROCK, PAPER, SCISSORS}}
	}

	push := func(rps RockPaperScissors, hands Hands) (RockPaperScissors, error) {
		return RockPaperScissors{Hand1: hands[0], Hand2: hands[1]}, nil
	}

	equal := func(rps1, rps2 *RockPaperScissors) bool {
		return *rps1 == *rps2
	}

	isEnd := func(rps *RockPaperScissors) bool {
		return rps.Hand1 != "" && rps.Hand2 != ""
	}

	game := simultaneous.Game[RockPaperScissors, Handss, Hands, Hand]{
		LegalActionss: legalActionss,
		Push:          push,
		Equal:         equal,
		IsEnd:         isEnd,
	}

	game.SetRandomActionPlayer(r)

	leafNodeEvalsFunc := func(rps *RockPaperScissors) duct.LeafNodeEvalYs {
		if rps.Hand1 == rps.Hand2 {
			return duct.LeafNodeEvalYs{0.5, 0.5}
		}

		reward := map[Hand]map[Hand]duct.LeafNodeEvalY{
			ROCK:     map[Hand]duct.LeafNodeEvalY{SCISSORS: 1.0, PAPER: 0.0},
			SCISSORS: map[Hand]duct.LeafNodeEvalY{ROCK: 0.0, PAPER: 1.0},
			PAPER:    map[Hand]duct.LeafNodeEvalY{ROCK: 1.0, SCISSORS: 0.0},
		}

		y := reward[rps.Hand1][rps.Hand2]
		return duct.LeafNodeEvalYs{y, 1.0 - y}
	}

	mcts := duct.MCTS[RockPaperScissors, Handss, Hands, Hand]{
		Game:              game,
		LeafNodeEvalsFunc: leafNodeEvalsFunc,
	}
	mcts.SetUniformActionPoliciesFunc()
	//mcts.UCBFunc = ucb.New1Func(math.Sqrt(1000))
	mcts.UCBFunc = ucb.NewAlphaGoFunc(math.Sqrt(2))

	fmt.Println(mcts.ActionPoliciesFunc(&RockPaperScissors{}))
	allNodes := mcts.NewAllNodes(&RockPaperScissors{})
	allNodes, err := mcts.Run(19600, allNodes, r)
	if err != nil {
		panic(err)
	}
	for i, m := range allNodes[0].UCBManagers {
		for a, pucb := range m {
			fmt.Println(i, a, pucb.AverageValue(), pucb.Trial)
		}
	}
	fmt.Println("")
	allNodes, err = mcts.Run(19600, allNodes, r)
	if err != nil {
		panic(err)
	}

	for i, m := range allNodes[0].UCBManagers {
		for a, pucb := range m {
			fmt.Println(i, a, pucb.AverageValue(), pucb.Trial)
		}
	}

	fmt.Println(allNodes[0].MaxTrialJointActionPath(r, 64))
}
