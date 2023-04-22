package crow_test

import (
	"testing"
	"fmt"
	"math"
	"github.com/sw965/omw"
	"github.com/sw965/crow"
	"golang.org/x/exp/slices"
)

func Test(t *testing.T) {
	r := omw.NewMt19937()

	legalActions := func(coins *[]int) []int {
		return omw.MakeIntegerRange[[]int](0, 10, 1)
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
		return len(*coins) == 5, reward / 45.0
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
	allNodes := puct.Run(19600, init, math.Sqrt(25.0), r)
	for a, pucb := range allNodes[0].PUCBManager {
		fmt.Println(a, pucb.AccumReward, pucb.Trial, pucb.AverageReward())
	}
}