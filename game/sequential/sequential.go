package sequential

import (
	"math/rand"
	"github.com/sw965/omw"
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

func (g *Game[S, AS, A]) SetRandomActionPlayer(r *rand.Rand) {
	g.Player = func(state *S) (A, error) {
		legals := g.LegalActions(state)
		ret := omw.RandChoice(legals, r)
		return ret, nil
	}
}

func (g *Game[S, AS, A]) Playout(state S) (S, error) {
	for {
		isEnd := g.IsEnd(&state)
		if isEnd {
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
	}
	return state, nil
}