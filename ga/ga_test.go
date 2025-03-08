package ga_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"github.com/sw965/crow/ga"
	"golang.org/x/exp/slices"
	omwrand "github.com/sw965/omw/math/rand"
)

func Test(t *testing.T) {
	r := omwrand.NewMt19937()

	popSize := 640
	individualLen := 10

	// 個体の型は []float64 として定義し、Population は [][]float64 となる
	initPop := make(ga.Population[[]float64], popSize)
	for i := 0; i < popSize; i++ {
		ind := make([]float64, individualLen)
		for j := 0; j < individualLen; j++ {
			ind[j] = omwrand.Float64(-10.0, 10.0, r)
		}
		initPop[i] = ind
	}

	// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] に近い個体が最適になる適応関数
	fitness := func(p ga.Population[[]float64], idx int) ga.FitnessY {
		sum := 0.0
		for i, gene := range p[idx] {
			sum += math.Abs(gene - float64(i))
		}
		// 損失が小さいほど適応度が高くなるように（最大値は 0）
		return ga.FitnessY(-1.0 * sum)
	}

	// 突然変異オペレーター: ランダムに1箇所の遺伝子を少し変化させる
	mutationOperator := func(ind []float64, r *rand.Rand) []float64 {
		mutated := make([]float64, len(ind))
		copy(mutated, ind)
		idx := r.Intn(len(mutated))
		if omwrand.Bool(r) {
			mutated[idx] += r.Float64()
		} else {
			mutated[idx] -= r.Float64()
		}
		return mutated
	}

	engine := ga.Engine[[]float64]{
		Fitness:          fitness,
		IndexSelector:    ga.NewLinearRankingIndexSelector(1.5),
		CrossOperator:    ga.UniformCrossOperator[float64],
		MutationOperator: mutationOperator,
		CloneOperator:    slices.Clone[[]float64, float64],
		CrossPercent:     0.5,
		MutationPercent:  0.01,
		EliteNum:         8,
	}

	generations := 1000
	finalPop, err := engine.Run(initPop, generations, r)
	if err != nil {
		t.Fatalf("Engine run failed: %v", err)
	}

	// 最も適応度の高い個体（エリート個体の1つ）を出力
	fmt.Println("Best individual:", finalPop[0])
}