package duct_test

import (
	"fmt"
	game "github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/game/solver"
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

type HandTable []Hands

type RockPaperScissors struct {
	Hand1 Hand
	Hand2 Hand
}

func TestDUCT(t *testing.T) {
	r := omwrand.NewMt19937()

	legalActionTableProvider := func(rps *RockPaperScissors) HandTable {
		return HandTable{HANDS, Hands{ROCK, PAPER, SCISSORS}}
	}

	transitioner := func(rps RockPaperScissors, hands Hands) (RockPaperScissors, error) {
		return RockPaperScissors{Hand1: hands[0], Hand2: hands[1]}, nil
	}

	comparator := func(rps1, rps2 *RockPaperScissors) bool {
		return *rps1 == *rps2
	}

	placementsJudger := func(rps *RockPaperScissors) (game.Placements, error) {
		if rps.Hand1 == rps.Hand2 {
			// 引き分けの場合は同順位
			return game.Placements{1, 1}, nil
		}
	
		hand1 := rps.Hand1
		hand2 := rps.Hand2
	
		if hand1 == ROCK {
			if hand2 == SCISSORS {
				return game.Placements{1, 2}, nil
			} else if hand2 == PAPER {
				return game.Placements{2, 1}, nil
			}
		}
	
		if hand1 == SCISSORS {
			if hand2 == ROCK {
				return game.Placements{2, 1}, nil
			} else if hand2 == PAPER {
				return game.Placements{1, 2}, nil
			}
		}
	
		if hand1 == PAPER {
			if hand2 == ROCK {
				return game.Placements{1, 2}, nil
			} else if hand2 == SCISSORS {
				return game.Placements{2, 1}, nil
			}
		}
	
		return game.Placements{}, nil
	}

	gameLogic := game.Logic[RockPaperScissors, HandTable, Hands, Hand]{
		LegalActionTableProvider: legalActionTableProvider,
		Transitioner:             transitioner,
		Comparator:               comparator,
		PlacementsJudger:         placementsJudger,
	}
	gameLogic.SetStandardResultScoresEvaluator()

	gameEngine := game.Engine[RockPaperScissors, HandTable, Hands, Hand]{
		Logic:gameLogic,
		Players:game.Players[RockPaperScissors, HandTable, Hands, Hand]{
			game.NewRandActionPlayer[RockPaperScissors, HandTable, Hands, Hand](r),
			game.NewRandActionPlayer[RockPaperScissors, HandTable, Hands, Hand](r),
		},
	}

	mcts := duct.Engine[RockPaperScissors, HandTable, Hands, Hand]{
		NextNodesCap:3,
	}
	mcts.SetGameEngine(gameEngine)

	mcts.PoliciesProvider = solver.UniformPoliciesProvider[RockPaperScissors, HandTable, Hands, Hand]
	mcts.UCBFunc = ucb.NewAlphaGoFunc(math.Sqrt(2))
	mcts.SetPlayout()

	fmt.Println(mcts.PoliciesProvider(&RockPaperScissors{}, legalActionTableProvider(&RockPaperScissors{})))
	rootNode, err := mcts.NewNode(&RockPaperScissors{})
	if err != nil {
		panic(err)
	}

	err = mcts.Run(19600, rootNode, r)
	if err != nil {
		panic(err)
	}

	for playerI, m := range rootNode.UCBManagers {
		for a, pucb := range m {
			fmt.Println("playerI =", playerI, a, pucb.AverageValue(), pucb.Trial)
		}
	}
	fmt.Println("")
	err = mcts.Run(19600, rootNode, r)
	if err != nil {
		panic(err)
	}

	for playerI, m := range rootNode.UCBManagers {
		for a, pucb := range m {
			fmt.Println("playerI =", playerI, a, pucb.AverageValue(), pucb.Trial)
		}
	}
}
