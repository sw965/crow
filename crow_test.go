package crow_test

import (
	"testing"
	"fmt"
	"math"
	"github.com/sw965/omw"
	"github.com/sw965/crow"
	"golang.org/x/exp/slices"
)

func TestPUCT(t *testing.T) {
	r := omw.NewMt19937()
	ALL_COINS := omw.MakeIntegerRange[[]int](0, 5, 1)
	MAX_COIN := omw.Max(ALL_COINS...)

	legalActions := func(coins *[]int) []int {
		return ALL_COINS
	}

	push := func(coins []int, coin int) []int {
		y := make([]int, 0, len(coins) + 1)
		for _, c := range coins {
			y = append(y, c)
		}
		y = append(y, coin)
		return y
	}

	equal := func(coins1, coins2 *[]int) bool {
		return slices.Equal(*coins1, *coins2)
	}

	isEndWithReward := func(coins *[]int) (bool, crow.Reward) {
		reward := crow.Reward(omw.Sum(*coins...))
		n := 10
		return len(*coins) == n, reward / crow.Reward(MAX_COIN * n)
	}

	game := crow.AlternatelyMoveGameFunCaller[[]int, int]{
		LegalActions:legalActions,
		Push:push,
		EqualState:equal,
		IsEndWithReward:isEndWithReward,
	}
	game.SetRandomActionPlayer(r)

	backward := func(y crow.PUCT_LeafEvalY, coins *[]int) crow.PUCT_BackwardEvalY {
		return crow.PUCT_BackwardEvalY(y)
	}
	eval := crow.PUCT_EvalFunCaller[[]int]{Backward:backward}

	fnCaller := crow.PUCT_FunCaller[[]int, int]{Game:game, Eval:eval}
	fnCaller.SetNoPolicy()
	fnCaller.SetPlayoutLeafEval()

	puct := crow.PUCT[[]int, int]{FunCaller:fnCaller}
	init := []int{}
	allNodes := puct.Run(196000, init, math.Sqrt(25.0), r)
	for _, coin := range ALL_COINS {
		pucb := allNodes[0].PUCBManager[coin]
		fmt.Println(coin, pucb.AccumReward, pucb.Trial, pucb.AverageReward())
	}

	fmt.Println(allNodes[0].ActionPrediction(r, 64))
}

type PPSState struct {
	Hand1 string
	Hand2 string
}

func TestDPUCT(t *testing.T) {
	r := omw.NewMt19937()
	var HANDS = []string{"グー", "チョキ", "パー"}

	legalActionss := func(state *PPSState) ([][]string) {
		return [][]string{HANDS, []string{"チョキ", "パー"}}
	}

	push := func(state PPSState, hands ...string) PPSState {
		return PPSState{Hand1:hands[0], Hand2:hands[1]}
	}

	equal := func(state1, state2 *PPSState) bool {
		return *state1 == *state2
	}

	isEndWithReward := func(state *PPSState) (bool, crow.Reward) {
		rewards := map[string]map[string]crow.Reward{
			"グー":map[string]crow.Reward{"チョキ":1.0, "パー":0.0},
			"チョキ":map[string]crow.Reward{"グー":0.0, "パー":1.0},
			"パー":map[string]crow.Reward{"グー":1.0, "チョキ":0.0},
		}
		if state.Hand1 == state.Hand2 {
			return true, 0.5
		} else {
			return true, rewards[state.Hand1][state.Hand2]
		}
	}

	game := crow.SimultaneousMoveGameFunCaller[PPSState, string]{
		LegalActionss:legalActionss,
		Push:push,
		EqualState:equal,
		IsEndWithReward:isEndWithReward,
	}

	game.SetRandomActionPlayer(r)

	leafEvals := func(state *PPSState) crow.DPUCT_LeafEvalYs {
		_, reward := isEndWithReward(state)
		leaf := crow.DPUCT_LeafEvalY(reward)
		return crow.DPUCT_LeafEvalYs{leaf, 1.0 - leaf}
	}

	fnCaller := crow.DPUCT_FunCaller[PPSState, string]{
		Game:game,
		LeafEvals:leafEvals,
	}
	fnCaller.SetNoPolicies()
	fmt.Println(fnCaller.Policies(&PPSState{}))

	dpuct := crow.DPUCT[PPSState, string]{FunCaller:fnCaller}
	allNodes := dpuct.Run(196000, PPSState{}, math.Sqrt(25), r)
	for i, m := range allNodes[0].PUCBManagers {
		for a, pucb := range m {
			fmt.Println(i, a, pucb.AverageReward(), pucb.Trial)
		}
	}

	fmt.Println(allNodes[0].ActionPrediction(r, 64))
}