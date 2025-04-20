package ga_test

import (
	"fmt"
	"math"
	"testing"
	"github.com/sw965/crow/ga"
	"golang.org/x/exp/slices"
	omwrand "github.com/sw965/omw/math/rand"
)

func Test(t *testing.T) {
	r := omwrand.NewMt19937()

	popSize := 640
	individualLen := 10

	// 個体の型は []float32 として定義し、Population は [][]float32 となる
	initPop := make(ga.Population[[]float32], popSize)
	for i := 0; i < popSize; i++ {
		ind := make([]float32, individualLen)
		for j := 0; j < individualLen; j++ {
			ind[j] = omwrand.Float64(-10.0, 10.0, r)
		}
		initPop[i] = ind
	}

	//明示しないと、SortPopulationが使えない
	var fitness ga.Fitness[[]float32]
	// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] に近い個体が最適になる適応関数
	fitness = func(p ga.Population[[]float32], idx int) ga.FitnessY {
		sum := 0.0
		for i, gene := range p[idx] {
			sum += math.Abs(gene - float32(i))
		}
		// 損失が小さいほど適応度が高くなるように（最大値は 0）
		return ga.FitnessY(-1.0 * sum)
	}

	engine := ga.Engine[[]float32]{
		Fitness:          fitness,
		IndexSelector:    ga.NewLinearRankingIndexSelector(1.5),
		CrossOperator:    ga.UniformCrossOperator[[]float32],
		MutationOperator: ga.NewUniformMutationOperator[[]float32](-5.0, 5.0),
		CloneOperator:    slices.Clone[[]float32, float32],
		CrossPercent:     0.5,
		MutationPercent:  0.01,
		EliteNum:         1,
	}

	generations := 100
	finalPop, err := engine.Run(initPop, generations, r)
	if err != nil {
		t.Fatalf("Engine run failed: %v", err)
	}

	sortedPopulation := fitness.SortPopulation(finalPop)
	for _, ind := range sortedPopulation {
		fmt.Println(ind)
	}
}