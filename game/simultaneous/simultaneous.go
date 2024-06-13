package simultaneous

import (
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
)

type Player[S any, AS ~[]A, A comparable] func(*S) (AS, []float64, error)
type LegalSeparateActionsFunc[S any, ASS ~[]AS, AS ~[]A, A comparable] func(*S) ASS
type PushFunc[S any, AS ~[]A, A comparable] func(S, AS) (S, error)
type EqualFunc[S any] func(*S, *S) bool
type IsEndFunc[S any] func(*S) (bool, []float64)

type Game[S any, ASS ~[]AS, AS ~[]A, A comparable] struct {
	Player Player[S, AS, A]
	LegalSeparateActions LegalSeparateActionsFunc[S, ASS, AS, A]
	Push PushFunc[S, AS, A]
	Equal EqualFunc[S]
	IsEnd IsEndFunc[S]
}

func (g *Game[S, ASS, AS, A]) Clone() Game[S, ASS, AS, A] {
	return Game[S, ASS, AS, A]{
		Player:g.Player,
		LegalSeparateActions:g.LegalSeparateActions,
		Push:g.Push,
		Equal:g.Equal,
		IsEnd:g.IsEnd,
	}
}

func (g *Game[S, ASS, AS, A]) SetRandActionPlayer(r *rand.Rand) {
	g.Player = func(state *S) (AS, []float64, error) {
		ass := g.LegalSeparateActions(state)
		ret := make(AS, len(ass))
		for playerI, as := range ass {
			ret[playerI] = omwrand.Choice(as, r)
		} 
		return ret, []float64{}, nil
	}
}

func (g *Game[S, ASS, AS, A]) Play(state S, f func(*S, int) bool) (S, error) {
	i := 0
	for {
		isEnd, _ := g.IsEnd(&state)
		if isEnd || f(&state, i) {
			break
		}

		jointAction, _, err := g.Player(&state)
		if err != nil {
			var s S
			return s, err
		}
		state, err = g.Push(state, jointAction)
		if err != nil {
			var s S
			return s, err
		}
		i += 1
	}
	return state, nil
}

func (g *Game[S, ASS, AS, A]) PlayWithHistory(state S, f func(*S, int) bool, c int) (S, []S, ASS, [][]float64, error) {
	i := 0
	stateHistory := make([]S, 0, c)
	jointActionHistory := make(ASS, 0, c)
	jointEvalHistory := make([][]float64, 0, c)

	for {
		isEnd, _ := g.IsEnd(&state)
		if isEnd || f(&state, i) {
			break
		}

		jointAction, jointEval, err := g.Player(&state)
		if err != nil {
			var s S
			return s, stateHistory, jointActionHistory, jointEvalHistory, err
		}

		stateHistory = append(stateHistory, state)
		jointActionHistory = append(jointActionHistory, jointAction)
		jointEvalHistory = append(jointEvalHistory, jointEval)

		state, err = g.Push(state, jointAction)
		if err != nil {
			var s S
			return s, stateHistory, jointActionHistory, jointEvalHistory, err
		}
		i += 1
	}
	return state, stateHistory, jointActionHistory, jointEvalHistory, nil
}

func (g *Game[S, ASS, AS, A]) Playout(state S) (S, error) {
	return g.Play(state, func(_ *S, _ int) bool { return false })
}

func (g *Game[S, ASS, AS, A]) PlayoutWithHistory(state S, capacity int) (S, []S, ASS, [][]float64, error) {
	return g.PlayWithHistory(state, func(_ *S, _ int) bool { return false }, capacity)
}