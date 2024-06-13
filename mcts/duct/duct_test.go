package duct_test

import (
	"fmt"
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/mcts/duct"
	"github.com/sw965/crow/ucb"
	omwrand "github.com/sw965/omw/math/rand"
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
	r := omwrand.NewMt19937()

	legalSeparateActions := func(rps *RockPaperScissors) Handss {
		return Handss{HANDS, Hands{ROCK, PAPER, SCISSORS}}
	}

	push := func(rps RockPaperScissors, hands Hands) (RockPaperScissors, error) {
		return RockPaperScissors{Hand1: hands[0], Hand2: hands[1]}, nil
	}

	equal := func(rps1, rps2 *RockPaperScissors) bool {
		return *rps1 == *rps2
	}

	isEnd := func(rps *RockPaperScissors) (bool, []float64) {
		isGameEnd := rps.Hand1 != "" && rps.Hand2 != ""
		reward := map[Hand]map[Hand]float64{
			ROCK:     map[Hand]float64{SCISSORS: 1.0, PAPER: 0.0},
			SCISSORS: map[Hand]float64{ROCK: 0.0, PAPER: 1.0},
			PAPER:    map[Hand]float64{ROCK: 1.0, SCISSORS: 0.0},
			"":       map[Hand]float64{"":-1.0},
		}
		p1Reward := reward[rps.Hand1][rps.Hand2]
		return isGameEnd, []float64{p1Reward, 1.0-p1Reward}
	}

	game := simultaneous.Game[RockPaperScissors, Handss, Hands, Hand]{
		LegalSeparateActions: legalSeparateActions,
		Push:          push,
		Equal:         equal,
		IsEnd:         isEnd,
	}

	game.SetRandActionPlayer(r)

	leafNodeJointEvalFunc := func(rps *RockPaperScissors) (duct.LeafNodeJointEvalY, error) {
		if rps.Hand1 == rps.Hand2 {
			return duct.LeafNodeJointEvalY{0.5, 0.5}, nil
		}

		reward := map[Hand]map[Hand]float64{
			ROCK:     map[Hand]float64{SCISSORS: 1.0, PAPER: 0.0},
			SCISSORS: map[Hand]float64{ROCK: 0.0, PAPER: 1.0},
			PAPER:    map[Hand]float64{ROCK: 1.0, SCISSORS: 0.0},
		}

		y := reward[rps.Hand1][rps.Hand2]
		return duct.LeafNodeJointEvalY{y, 1.0 - y}, nil
	}

	mcts := duct.MCTS[RockPaperScissors, Handss, Hands, Hand]{
		Game:              game,
		LeafNodeJointEvalFunc: leafNodeJointEvalFunc,
	}

	mcts.SetUniformSeparateActionPolicyFunc()
	//mcts.UCBFunc = ucb.New1Func(math.Sqrt(1000))
	mcts.UCBFunc = ucb.NewAlphaGoFunc(math.Sqrt(2))

	fmt.Println(mcts.SeparateActionPolicyFunc(&RockPaperScissors{}))
	rootNode := mcts.NewNode(&RockPaperScissors{})
	err := mcts.Run(19600, rootNode, r)
	if err != nil {
		panic(err)
	}
	for i, m := range rootNode.SeparateUCBManager {
		for a, pucb := range m {
			fmt.Println(i, a, pucb.AverageValue(), pucb.Trial)
		}
	}
	fmt.Println("")
	err = mcts.Run(19600, rootNode, r)
	if err != nil {
		panic(err)
	}

	for i, m := range rootNode.SeparateUCBManager {
		for a, pucb := range m {
			fmt.Println(i, a, pucb.AverageValue(), pucb.Trial)
		}
	}

	fmt.Println(rootNode.Predict(r, 64))
}
