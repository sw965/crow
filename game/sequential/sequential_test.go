package sequential_test

import (
	"testing"
	"github.com/sw965/crow/game/sequential"
	"fmt"
)

type State string
type Agent string

func Test(t *testing.T) {
	logic := sequential.Logic[State, []string, string, Agent]{}
	logic.PlacementsJudger = func(_ *State) (sequential.PlacementByAgent[Agent], error) {
		return sequential.NewPlacementByAgent[[][]Agent, []Agent, Agent](
			[][]Agent{[]Agent{"白"}, []Agent{"水"}},
		)
	}
	logic.SetStandardResultScoresEvaluator()

	var s State
	fmt.Println(logic.EvaluateResultScoreByAgent(&s))
}