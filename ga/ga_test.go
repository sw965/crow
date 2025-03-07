package ga_test

import (
	"testing"
	"github.com/sw965/crow/ga" // 実際の環境に合わせて適切なインポートパスに変更してください
	"fmt"
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
)

func Test(t *testing.T) {
	r := omwrand.NewMt19937()

	popSize := 20
	individualLen := 10

	initPop := make(ga.Population[int], popSize)
	for i := 0; i < popSize; i++ {
		ind := make(ga.Individual[int], individualLen)
		for j := 0; j < individualLen; j++ {
			if r.Float64() < 0.5 {
				ind[j] = 0
			} else {
				ind[j] = 1
			}
		}
		initPop[i] = ind
	}

	fitness := func(p ga.Population[int], idx int) ga.FitnessY {
		sum := 0
		for _, gene := range p[idx] {
			sum += gene
		}
		return ga.FitnessY(sum)
	}

	idxSelector := ga.RouletteIndexSelector
	crossOperator := ga.UniformCrossOperator[int]

	mutationOperator := func(ind ga.Individual[int], r *rand.Rand) ga.Individual[int] {
		mutated := make(ga.Individual[int], len(ind))
		copy(mutated, ind)
		pos := r.Intn(len(mutated))
		if mutated[pos] == 0 {
			mutated[pos] = 1
		} else {
			mutated[pos] = 0
		}
		return mutated
	}

	engine := ga.Engine[int]{
		Fitness:          fitness,
		IndexSelector:    idxSelector,
		CrossOperator:    crossOperator,
		MutationOperator: mutationOperator,
		CrossPercent:     0.5,
		MutationPercent:  0.01,
	}

	generations := 500
	finalPop, err := engine.Run(initPop, generations, r)
	if err != nil {
		panic(err)
	}

	fmt.Println(finalPop)
}