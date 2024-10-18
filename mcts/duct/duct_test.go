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

	separateLegalActionsProvider := func(rps *RockPaperScissors) Handss {
		return Handss{HANDS, Hands{ROCK, PAPER, SCISSORS}}
	}

	transitioner := func(rps RockPaperScissors, hands Hands) (RockPaperScissors, error) {
		return RockPaperScissors{Hand1: hands[0], Hand2: hands[1]}, nil
	}

	comparator := func(rps1, rps2 *RockPaperScissors) bool {
		return *rps1 == *rps2
	}

	endChecker := func(rps *RockPaperScissors) bool {
		isGameEnd := rps.Hand1 != "" && rps.Hand2 != ""
		return isGameEnd
	}

	gameLogic := simultaneous.Logic[RockPaperScissors, Handss, Hands, Hand]{
		SeparateLegalActionsProvider: separateLegalActionsProvider,
		Transitioner:                 transitioner,
		Comparator:                   comparator,
		EndChecker:                   endChecker,
	}

	leafNodeJointEvaluator := func(rps *RockPaperScissors) (duct.LeafNodeJointEval, error) {
		if rps.Hand1 == rps.Hand2 {
			return duct.LeafNodeJointEval{0.5, 0.5}, nil
		}

		reward := map[Hand]map[Hand]float64{
			ROCK:     map[Hand]float64{SCISSORS: 1.0, PAPER: 0.0},
			SCISSORS: map[Hand]float64{ROCK: 0.0, PAPER: 1.0},
			PAPER:    map[Hand]float64{ROCK: 1.0, SCISSORS: 0.0},
		}

		y := reward[rps.Hand1][rps.Hand2]
		return duct.LeafNodeJointEval{y, 1.0 - y}, nil
	}

	mcts := duct.MCTS[RockPaperScissors, Handss, Hands, Hand]{
		GameLogic:              gameLogic,
		LeafNodeJointEvaluator: leafNodeJointEvaluator,
		NextNodesCap:           3,
	}

	mcts.SetUniformSeparateActionPolicyProvider()
	mcts.UCBFunc = ucb.NewAlphaGoFunc(math.Sqrt(2))

	fmt.Println(mcts.SeparateActionPolicyProvider(&RockPaperScissors{}))
	rootNode := mcts.NewNode(&RockPaperScissors{})
	err := mcts.Run(19600, rootNode, r)
	if err != nil {
		panic(err)
	}
	for playerI, m := range rootNode.SeparateUCBManager {
		for a, pucb := range m {
			fmt.Println("playerI =", playerI, a, pucb.AverageValue(), pucb.Trial)
		}
	}
	fmt.Println("")
	err = mcts.Run(19600, rootNode, r)
	if err != nil {
		panic(err)
	}

	for playerI, m := range rootNode.SeparateUCBManager {
		for a, pucb := range m {
			fmt.Println("playerI =", playerI, a, pucb.AverageValue(), pucb.Trial)
		}
	}
}
