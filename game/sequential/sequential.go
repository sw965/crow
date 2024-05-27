package sequential

import (
	"math/rand"
	omwrand "github.com/sw965/omw/math/rand"
)

type Player[S any, A comparable] func(*S) (A, error)
type LegalActionsFunc[S any, AS ~[]A, A comparable] func(*S) AS
type PushFunc[S any, A comparable] func(S, *A) (S, error)
type EqualFunc[S any] func(*S, *S) bool
type IsEndFunc[S any] func(*S) bool

type Game[S any, AS ~[]A, A comparable] struct {
	Player Player[S, A]
	LegalActions LegalActionsFunc[S, AS, A]
	Push PushFunc[S, A]
	Equal EqualFunc[S]
	IsEnd IsEndFunc[S]
}

func (g *Game[S, AS, A]) Clone() Game[S, AS, A] {
	return Game[S, AS, A]{
		Player:g.Player,
		LegalActions:g.LegalActions,
		Push:g.Push,
		Equal:g.Equal,
		IsEnd:g.IsEnd,
	}
}

func (g *Game[S, AS, A]) SetRandActionPlayer(r *rand.Rand) {
	g.Player = func(state *S) (A, error) {
		as := g.LegalActions(state)
		return omwrand.Choice(as, r), nil
	}
}

func (g *Game[S, AS, A]) Play(state S, f func(*S, int) bool) (S, error) {
	i := 0
	for {
		if g.IsEnd(&state) || f(&state, i) {
			break
		}

		action, err := g.Player(&state)
		if err != nil {
			var s S
			return s, err
		}

		state, err = g.Push(state, &action)
		if err != nil {
			var s S
			return s, err
		}
		i += 1
	}
	return state, nil
}

func (g *Game[S, AS, A]) PlayWithHistory(state S, f func(*S, int) bool, c int) (S, []S, AS, error) {
	i := 0
	stateHistory := make([]S, 0, c)
	actionHistory := make(AS, 0, c)
	for {
		if g.IsEnd(&state) || f(&state, i) {
			break
		}

		action, err := g.Player(&state)
		if err != nil {
			var s S
			return s, []S{}, AS{}, err
		}

		stateHistory = append(stateHistory, state)
		actionHistory = append(actionHistory, action)

		state, err = g.Push(state, &action)
		if err != nil {
			var s S
			return s, []S{}, AS{}, err
		}
		i += 1
	}
	return state, stateHistory, actionHistory, nil
}

func (g *Game[S, AS, A]) Playout(state S) (S, error) {
	return g.Play(state, func(_ *S, _ int) bool { return false})
}

func (g *Game[S, AS, A]) PlayoutWithHistory(state S, c int) (S, []S, AS, error) {
	return g.PlayWithHistory(state, func(_ *S, _ int) bool { return false}, c)
}