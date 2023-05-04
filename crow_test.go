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
	COIN_TOSS_NUM := 10
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

	isEnd := func(coins *[]int) bool {
		return len(*coins) == COIN_TOSS_NUM
	}

	game := crow.SequentialGameFunCaller[[]int, []int, int]{
		LegalActions:legalActions,
		Push:push,
		EqualState:equal,
		IsEnd:isEnd,
	}
	game.SetRandomActionPlayer(r)

	leaf := func(coins *[]int) crow.PUCT_LeafEvalY {
		endCoins := game.Playout(*coins)
		sum := crow.PUCT_LeafEvalY(omw.Sum(endCoins...))
		return sum / crow.PUCT_LeafEvalY(MAX_COIN * COIN_TOSS_NUM)
	}

	backward := func(y crow.PUCT_LeafEvalY, coins *[]int) crow.PUCT_BackwardEvalY {
		return crow.PUCT_BackwardEvalY(y)
	}

	eval := crow.PUCT_EvalFunCaller[[]int]{Leaf:leaf, Backward:backward}

	fnCaller := crow.PUCT_FunCaller[[]int, []int, int]{Game:game, Eval:eval}
	fnCaller.SetNoPolicy()

	puct := crow.PUCT[[]int, []int, int]{FunCaller:fnCaller}
	init := []int{}
	allNodes := puct.Run(196000, init, math.Sqrt(25.0), r)
	for _, coin := range ALL_COINS {
		pucb := allNodes[0].PUCBManager[coin]
		fmt.Println(coin, pucb.AccumReward, pucb.Trial, pucb.AverageReward())
	}

	fmt.Println(allNodes[0].ActionPrediction(r, 64))
}