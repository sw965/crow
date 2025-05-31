package rl

import (
	"fmt"
	"math/rand"
	"slices"
	//cmath "github.com/sw965/crow/math"
	orand "github.com/sw965/omw/math/rand"
	game "github.com/sw965/crow/game/sequential"
)

type Experience[S any] struct {
	State       S
	ActionIndex int
	Reward      float32
}

type Experiences[S any] []Experience[S]

type Reinforcer[M, S any, As ~[]A, A, G comparable] struct {
	GameLogic    game.Logic[S, As, A, G]
	Predictor    func(M, S) ([]float32, error)
	ModelByAgent map[G]M
	Actions      As
}

func (r Reinforcer[M, S, As, A, G]) CollectExperiences(initStates []S, rngs []*rand.Rand) (Experiences[S], error) {
	player := func(state S, legalActions As, workerIdx int) (A, error) {
		agent := r.GameLogic.CurrentAgentGetter(state)
		model, ok := r.ModelByAgent[agent]
		if !ok {
			var a A
			return a, fmt.Errorf("エージェントが足りない")
		}

		y, err := r.Predictor(model, state)
		if err != nil {
			var a A
			return a, err
		}

		legalN := len(legalActions)
		legalActionIdxs := make([]int, legalN)
		legalY := make([]float32, legalN)

		for i, a := range legalActions {
			idx := slices.Index(r.Actions, a)
			legalActionIdxs[i] = idx
			legalY[i] = y[idx]
		}

		rng := rngs[workerIdx]
		idx := orand.IntByWeight(legalY, rng)
		actionIdx := legalActionIdxs[idx]
		action := r.Actions[actionIdx]
		return action, nil
	}

	p := len(rngs)
	history, err := r.GameLogic.PlayoutsWithHistory(initStates, player, p)
	if err != nil {
		return nil, err
	}

	n := len(history.ActionsByGame) * game.GetOneGameCap()

	experiences := make(Experiences[S], 0, n)
	for gameI, states := range history.IntermediateStatesByGame {
		final := history.FinalStateByGame[gameI]
		actions := history.ActionsByGame[gameI] 
		scores, err := r.GameLogic.EvaluateResultScoreByAgent(final)
		if err != nil {
			return nil, err
		}
		for i, state := range states {
			agent := r.GameLogic.CurrentAgentGetter(state)
			experience := Experience[S]{
				State:state,
				ActionIndex:slices.Index(r.Actions, actions[i]),
				Reward:scores[agent],
			}
			experiences = append(experiences, experience)
		}
	}
	return experiences, nil
}