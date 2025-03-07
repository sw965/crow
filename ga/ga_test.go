package ga_test

import (
	"testing"
	"github.com/sw965/crow/ga" // 実際の環境に合わせて適切なインポートパスに変更してください
	"fmt"
	"math"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
)

func Test(t *testing.T) {
	r := omwrand.NewMt19937()

	popSize := 640
	individualLen := 10

	initPop := make(ga.Population[float64], popSize)
	for i := 0; i < popSize; i++ {
		ind := make(ga.Individual[float64], individualLen)
		for j := range ind {
			ind[j] = omwrand.Float64(-10.0, 10.0, r)
		}
		initPop[i] = ind
	}

	// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] の重みが最適なパラメーターとなるような適応関数。
	fitness := func(p ga.Population[float64], idx int) ga.FitnessY {
		sum := 0.0
		for i, gene := range p[idx] {
			//期待する値との差が大きいほど、損失が大きくなるように。
			sum += math.Abs(gene - float64(i))
		}
		/*
			適応関数は、損失関数と違い、いいモデルである程、値が大きくなるようにしなければならないので、
			マイナスを掛けて符号を逆にする。
			この適応関数の場合、0が最大値(最も適応した状態)となる。
		*/
		return ga.FitnessY(-1.0 * sum)
	}

	mutationOperator := func(ind ga.Individual[float64], r *rand.Rand) ga.Individual[float64] {
		mutated := make(ga.Individual[float64], len(ind))
		copy(mutated, ind)
		idx := r.Intn(len(mutated))
		if omwrand.Bool(r) {
			mutated[idx] += r.Float64()
		} else {
			mutated[idx] -= r.Float64()
		}
		return mutated
	}

	engine := ga.Engine[float64]{
		Fitness:          fitness,
		IndexSelector:    ga.NewLinearRankingIndexSelector(1.5),
		CrossOperator:    ga.UniformCrossOperator[float64],
		MutationOperator: mutationOperator,
		CrossPercent:     0.5,
		MutationPercent:  0.01,
		EliteNum:8,
	}

	generations := 1000
	finalPop, err := engine.Run(initPop, generations, r)
	if err != nil {
		panic(err)
	}

	fmt.Println(finalPop[0])
}