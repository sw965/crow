package dpuct_test

import (
	"fmt"
	"github.com/sw965/crow/game/simultaneous"
	"github.com/sw965/crow/mcts/dpuct"
	omwrand "github.com/sw965/omw/rand"
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

func TestDPUCT(t *testing.T) {
	r := omwrand.NewMt19937()

	legalActionss := func(rps *RockPaperScissors) Handss {
		return Handss{HANDS, Hands{SCISSORS}}
	}

	push := func(rps RockPaperScissors, hands ...Hand) RockPaperScissors {
		return RockPaperScissors{Hand1: hands[0], Hand2: hands[1]}
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

	game.SetRandomActionPlayer(r, "")

	leafEvals := func(rps *RockPaperScissors) dpuct.LeafEvalYs {
		if rps.Hand1 == rps.Hand2 {
			return dpuct.LeafEvalYs{0.5, 0.5}
		}

		reward := map[Hand]map[Hand]dpuct.LeafEvalY{
			ROCK:     map[Hand]dpuct.LeafEvalY{SCISSORS: 1.0, PAPER: 0.0},
			SCISSORS: map[Hand]dpuct.LeafEvalY{ROCK: 0.0, PAPER: 1.0},
			PAPER:    map[Hand]dpuct.LeafEvalY{ROCK: 1.0, SCISSORS: 0.0},
		}

		y := reward[rps.Hand1][rps.Hand2]
		return dpuct.LeafEvalYs{y, 1.0 - y}
	}

	mcts := dpuct.MCTS[RockPaperScissors, Handss, Hands, Hand]{
		Game:      game,
		LeafEvals: leafEvals,
	}
	mcts.SetActionNoPolicies("")

	fmt.Println(mcts.ActionPolicies(&RockPaperScissors{}))

	allNodes := mcts.Run(196000, RockPaperScissors{}, math.Sqrt(25), r)
	for i, m := range allNodes[0].PUCBManagers {
		for a, pucb := range m {
			fmt.Println(i, a, pucb.AverageReward(), pucb.Trial)
		}
	}

	fmt.Println(allNodes[0].ActionPrediction(r, 64))
}
